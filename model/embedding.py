"""
model/embedding.py
==================
Etapa 7 do pipeline: Temporal Patch Embedding.

Motivação
---------
Em vez de processar cada passo de tempo individualmente (como no
Transformer original), o Temporal Patch Embedding agrupa passos
consecutivos em "patches", reduzindo o comprimento da sequência e
capturando padrões locais de curto prazo antes da atenção global.

Esta abordagem é inspirada no PatchTST (Nie et al., 2023), que adaptou
o Vision Transformer (ViT) para séries temporais, mostrando ganhos
substanciais sobre métodos anteriores em benchmarks de previsão.

Com WINDOW=168 e PATCH_SIZE=6:
  - Número de patches = 168 // 6 = 28
  - Cada patch cobre 6 horas (4 patches por dia, alinhado ao ciclo
    meteorológico de manhã/tarde/noite/madrugada)

Referências:
  - Nie, Y. et al. (2023). A Time Series is Worth 64 Words:
    Long-Term Forecasting with Transformers. ICLR 2023.
    arXiv:2211.14730
  - Dosovitskiy, A. et al. (2021). An image is worth 16x16 words:
    Transformers for image recognition at scale. ICLR 2021.
"""

import math

import torch
import torch.nn as nn


class TemporalPatchEmbedding(nn.Module):
    """
    Divide a janela temporal em patches não sobrepostos e os projeta
    para o espaço de dimensão d_model.

    Fluxo:
      (B, L, F) → reshape → (B, P, patch_size * F) → Linear → (B, P, d_model)
      → soma de embeddings posicionais aprendíveis → (B, P, d_model)

    onde:
      B = batch size
      L = comprimento da janela (WINDOW)
      F = número de features (INPUT_DIM)
      P = número de patches = L // patch_size

    Parâmetros
    ----------
    seq_len    : int  — comprimento da janela de entrada (L)
    n_features : int  — número de features (F)
    patch_size : int  — número de passos por patch
    d_model    : int  — dimensão do espaço de embedding
    dropout    : float — dropout aplicado após embedding posicional
    """

    def __init__(
        self,
        seq_len:    int,
        n_features: int,
        patch_size: int,
        d_model:    int,
        dropout:    float = 0.1,
    ):
        super().__init__()

        assert seq_len % patch_size == 0, (
            f"seq_len ({seq_len}) deve ser divisível por patch_size ({patch_size})"
        )

        self.patch_size  = patch_size
        self.num_patches = seq_len // patch_size      # P = 168 // 6 = 28
        patch_dim        = patch_size * n_features    # dimensão bruta de um patch

        # Projeção linear do patch para o espaço d_model
        # Equivalente a um "tokenizador" que mapeia cada patch para um vetor
        self.patch_projection = nn.Linear(patch_dim, d_model)

        # Embeddings posicionais aprendíveis — um vetor por posição de patch
    
        # Embeddings aprendíveis são preferíveis ao invés de senoidais fixos
        # porque a série temporal tem padrões periódicos que podem beneficiar
        # de representações posicionais ajustadas ao domínio (Nie et al., 2023)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )

        # Dropout aplicado após a soma do embedding posicional
        self.dropout = nn.Dropout(dropout)

        # Inicialização dos pesos
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Inicialização de pesos seguindo a prática do ViT:
          - Projeção linear: inicialização Xavier uniforme
          - Embeddings posicionais: distribuição normal truncada
        """
        nn.init.xavier_uniform_(self.patch_projection.weight)
        nn.init.zeros_(self.patch_projection.bias)
        # Embeddings posicionais com desvio pequeno para suavidade inicial
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parâmetros
        ----------
        x : Tensor, shape (B, L, F)

        Retorna
        -------
        Tensor, shape (B, P, d_model)
        """
        B, L, F = x.shape

        # 1. Reshape: (B, L, F) → (B, P, patch_size * F)
        #    Agrupa patch_size passos consecutivos em um vetor flatten
        x = x.reshape(B, self.num_patches, self.patch_size * F)

        # 2. Projeção linear: (B, P, patch_size*F) → (B, P, d_model)
        x = self.patch_projection(x)

        # 3. Soma dos embeddings posicionais aprendíveis
        x = x + self.pos_embedding

        # 4. Dropout
        x = self.dropout(x)

        return x  # (B, P, d_model)
