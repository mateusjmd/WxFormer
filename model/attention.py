"""
model/attention.py
==================
Camada de atenção customizada com Temporal Decay Bias (ALiBi-style)
e dropouts separados para atenção e feedforward.

Temporal Decay Bias
-------------------
Inspirado no ALiBi (Press et al., 2022), adicionamos um viés aditivo
ao mapa de atenção que penaliza pares de posições distantes:

    score(i, j) = (Q_i · K_j) / sqrt(d_k) + bias(i, j)

onde:
    bias(i, j) = -m_h · |i - j|

O slope m_h é específico por cabeça e segue a geometria:
    m_h = 2^(-8·h / n_heads),   h = 1, ..., n_heads

Isso induz um prior de localidade: tokens próximos recebem atenção
maior, refletindo a dependência temporal forte em séries meteorológicas
de curto prazo. Diferentes heads têm diferentes alcances efetivos,
permitindo que o modelo aprenda tanto dependências locais quanto globais.

Pre-Layer Normalization (Pre-LN)
---------------------------------
Aplicamos LayerNorm ANTES de cada sub-camada (Pre-LN), estratégia que
melhora a estabilidade do gradiente em comparação ao Post-LN do
Transformer original (Xiong et al., 2020).

Referências:
  - Press, O. et al. (2022). Train Short, Test Long: Attention with
    Linear Biases Enables Input Length Extrapolation. ICLR 2022.
    arXiv:2108.12409
  - Xiong, R. et al. (2020). On Layer Normalization in the Transformer
    Architecture. ICML 2020. arXiv:2002.04745
  - Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 2017.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Temporal Decay Bias (ALiBi)
# ---------------------------------------------------------------------------

def compute_alibi_bias(
    n_heads: int,
    seq_len: int,
    device:  torch.device,
) -> torch.Tensor:
    """
    Calcula o viés de decaimento temporal ALiBi para todas as cabeças.

    Para cada cabeça h:
        m_h   = 2^(-8·h / n_heads)
        B[h, i, j] = -m_h · |i - j|

    O viés é negativo (penalidade de distância) e diferenciado por cabeça.

    Parâmetros
    ----------
    n_heads : int           — número de cabeças de atenção
    seq_len : int           — comprimento da sequência (número de patches)
    device  : torch.device  — dispositivo de computação

    Retorna
    -------
    Tensor, shape (n_heads, seq_len, seq_len)
        Adicionado aos logits de atenção ANTES do softmax.
    """
    # Slopes: m_h = 2^(-8h/n_heads) para h = 1, ..., n_heads
    h_idx  = torch.arange(1, n_heads + 1, dtype=torch.float32, device=device)
    slopes = torch.pow(2.0, -8.0 * h_idx / n_heads)  # (n_heads,)

    # Matriz de distâncias absolutas: |i - j|
    positions  = torch.arange(seq_len, dtype=torch.float32, device=device)
    # (seq_len, seq_len): distâncias absolutas entre todas as posições
    dist_matrix = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()

    # Viés: (n_heads, seq_len, seq_len)  ← broadcast de slopes sobre dist
    # slopes[:, None, None] * dist_matrix[None, :, :] → negativo = penalidade
    bias = -slopes[:, None, None] * dist_matrix[None, :, :]

    return bias  # (n_heads, seq_len, seq_len)


# ---------------------------------------------------------------------------
# Camada Transformer Customizada
# ---------------------------------------------------------------------------

class CustomTransformerEncoderLayer(nn.Module):
    """
    Camada de encoder Transformer com:
      - Dropouts separados para atenção e feedforward
      - Temporal Decay Bias (ALiBi) no mecanismo de atenção
      - Pre-LN (LayerNorm antes de cada sub-camada)
      - Ativação GELU na rede feedforward

    Parâmetros
    ----------
    d_model      : int   — dimensão do modelo
    n_heads      : int   — número de cabeças de atenção
    dim_ff       : int   — dimensão interna da MLP feedforward
    attn_dropout : float — dropout aplicado nos pesos de atenção
    ff_dropout   : float — dropout aplicado na camada FF intermediária
    use_alibi    : bool  — se True, adiciona Temporal Decay Bias
    """

    def __init__(
        self,
        d_model:      int,
        n_heads:      int,
        dim_ff:       int,
        attn_dropout: float = 0.1,
        ff_dropout:   float = 0.1,
        use_alibi:    bool  = True,
    ):
        super().__init__()

        self.n_heads   = n_heads
        self.use_alibi = use_alibi

        # ---- Mecanismo de Auto-Atenção Multi-Cabeça -------------------
        # Dropout de atenção é passado diretamente ao MHA do PyTorch,
        # que o aplica internamente sobre os pesos softmax
        self.self_attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = attn_dropout,
            batch_first = True,   # (B, L, D) em vez de (L, B, D)
        )

        # ---- Rede Feedforward Posição-a-Posição ------------------------
        # Estrutura: Linear → GELU → Dropout → Linear
        # GELU é preferida ao ReLU em Transformers pré-treinados (Hendrycks
        # & Gimpel, 2016) por sua suavidade e melhor gradiente próximo a zero
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim_ff, d_model),
        )

        # ---- Normalização (Pre-LN) ------------------------------------
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # ---- Dropout residual após cada sub-camada --------------------
        # Nota: attn_dropout já atua internamente no MHA; este dropout
        # residual é aplicado à saída da atenção antes da soma residual
        self.dropout_attn = nn.Dropout(attn_dropout)
        self.dropout_ff   = nn.Dropout(ff_dropout)

        # Inicialização dos pesos da camada FF
        self._init_ff_weights()

    def _init_ff_weights(self) -> None:
        """Inicializa pesos lineares com Xavier uniforme."""
        for layer in self.ff:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x:         torch.Tensor,
        alibi_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Passagem adiante de uma camada do encoder.

        Parâmetros
        ----------
        x          : Tensor, shape (B, P, d_model)
        alibi_bias : Tensor, shape (n_heads, P, P) — opcional

        Retorna
        -------
        Tensor, shape (B, P, d_model)
        """
        # ------ Sub-camada 1: Atenção (Pre-LN) -------------------------
        residual = x
        x_normed = self.norm1(x)

        # Prepara a máscara de atenção para o MHA do PyTorch
        # MHA espera attn_mask: (P, P) ou (B*n_heads, P, P)
        attn_mask = None
        if self.use_alibi and alibi_bias is not None:
            B, P, _ = x.shape
            # Expande o bias de (n_heads, P, P) → (B*n_heads, P, P)
            attn_mask = alibi_bias.unsqueeze(0).expand(B, -1, -1, -1)
            attn_mask = attn_mask.reshape(B * self.n_heads, P, P)

        attn_out, _ = self.self_attn(
            x_normed, x_normed, x_normed,
            attn_mask   = attn_mask,
            need_weights = False,  # não retorna pesos aqui (veja explain.py)
        )
        x = residual + self.dropout_attn(attn_out)

        # ------ Sub-camada 2: Feedforward (Pre-LN) ---------------------
        residual = x
        x = residual + self.dropout_ff(self.ff(self.norm2(x)))

        return x
