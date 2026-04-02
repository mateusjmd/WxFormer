"""
model/transformer.py
====================
Etapas 7, 8 e 9 do pipeline: arquitetura completa do WeatherTransformer.

Arquitetura
-----------
  Input  (B, L, F)
     ↓
  TemporalPatchEmbedding  →  (B, P, d_model)
     ↓
  [ALiBi bias: (n_heads, P, P)]
     ↓
  N × CustomTransformerEncoderLayer
     ↓
  Mean pooling sobre patches  →  (B, d_model)
     ↓
  LayerNorm
     ↓
  Linear(d_model, horizon)   →  (B, horizon)

O uso de mean pooling (ao invés do último token) reduz a variância
da representação final e é menos sensível a artefatos do último patch
(Nie et al., 2023).

Referências:
  - Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 2017.
  - Nie, Y. et al. (2023). A Time Series is Worth 64 Words. ICLR 2023.
  - Press, O. et al. (2022). ALiBi. ICLR 2022.
"""

import torch
import torch.nn as nn

from model.attention import CustomTransformerEncoderLayer, compute_alibi_bias
from model.embedding import TemporalPatchEmbedding


class WeatherTransformer(nn.Module):
    """
    Transformer para previsão meteorológica multi-step com:
      - Temporal Patch Embedding
      - Temporal Decay Bias (ALiBi)
      - Dropouts separados por sub-camada
      - Pre-LN (Pre-Layer Normalization)
      - Cabeça de regressão linear

    Parâmetros
    ----------
    input_dim    : int   — número de features de entrada (F)
    d_model      : int   — dimensão do espaço de embedding
    n_heads      : int   — número de cabeças de atenção
    n_layers     : int   — número de camadas do encoder
    dim_ff       : int   — dimensão interna da MLP feedforward
    attn_dropout : float — dropout nos pesos de atenção
    ff_dropout   : float — dropout na camada FF
    horizon      : int   — número de passos a prever
    seq_len      : int   — comprimento da janela de entrada (L)
    patch_size   : int   — tamanho de cada patch temporal
    use_alibi    : bool  — se True, usa Temporal Decay Bias
    """

    def __init__(
        self,
        input_dim:    int   = 12,
        d_model:      int   = 256,
        n_heads:      int   = 4,
        n_layers:     int   = 4,
        dim_ff:       int   = 1024,
        attn_dropout: float = 0.1,
        ff_dropout:   float = 0.1,
        horizon:      int   = 24,
        seq_len:      int   = 168,
        patch_size:   int   = 6,
        use_alibi:    bool  = True,
    ):
        super().__init__()

        # Valida restrição entre d_model e n_heads
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) deve ser divisível por n_heads ({n_heads})"
        )

        self.horizon     = horizon
        self.use_alibi   = use_alibi
        self.num_patches = seq_len // patch_size

        # ---- 1. Temporal Patch Embedding -----------------------------
        self.patch_embed = TemporalPatchEmbedding(
            seq_len    = seq_len,
            n_features = input_dim,
            patch_size = patch_size,
            d_model    = d_model,
            dropout    = ff_dropout,  # mesmo dropout do FF na camada de embed
        )

        # ---- 2. Stack de camadas do encoder --------------------------
        # Cada camada tem seus próprios pesos; o bias ALiBi é
        # recalculado a cada forward pass (custo negligível, O(P²·H))
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model      = d_model,
                n_heads      = n_heads,
                dim_ff       = dim_ff,
                attn_dropout = attn_dropout,
                ff_dropout   = ff_dropout,
                use_alibi    = use_alibi,
            )
            for _ in range(n_layers)
        ])

        # ---- 3. Normalização final (após mean pooling) ---------------
        self.final_norm = nn.LayerNorm(d_model)

        # ---- 4. Cabeça de regressão ----------------------------------
        # Projeção direta de d_model → horizon (previsão multi-step)
        self.head = nn.Linear(d_model, horizon)

        # Inicializa pesos da cabeça com escala pequena
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        """
        Passagem adiante do modelo.

        Parâmetros
        ----------
        x           : Tensor, shape (B, L, F)
        return_attn : bool — se True, retorna também os pesos de atenção
                             (usado em explainability/explain.py)

        Retorna
        -------
        out : Tensor, shape (B, horizon)  — previsões normalizadas
        [attn_weights] : list de Tensors se return_attn=True
        """
        # 1. Patch Embedding: (B, L, F) → (B, P, d_model)
        x = self.patch_embed(x)

        # 2. Calcula o bias ALiBi (shape: n_heads, P, P)
        #    Calculado aqui (não em __init__) para corrigir o device automaticamente
        alibi_bias = None
        if self.use_alibi:
            alibi_bias = compute_alibi_bias(
                n_heads = self.encoder_layers[0].n_heads,
                seq_len = self.num_patches,
                device  = x.device,
            )

        # 3. Passa pelo stack de camadas do encoder
        attn_weights_list = []  # lista para retornar pesos de atenção
        for layer in self.encoder_layers:
            x = layer(x, alibi_bias=alibi_bias)

        # 4. Mean Pooling: (B, P, d_model) → (B, d_model)
        #    Agrega informação de todos os patches (robusto a bordas)
        x = x.mean(dim=1)

        # 5. Normalização final
        x = self.final_norm(x)

        # 6. Cabeça de regressão: (B, d_model) → (B, horizon)
        out = self.head(x)

        return out

    def count_parameters(self) -> int:
        """Retorna o número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
