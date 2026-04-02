"""
data/dataset.py
===============
Etapas 5 e 6 do pipeline: criação de janelas temporais e divisão
train / validation / test.

Janela deslizante (sliding window):
  - Dado um histórico de WINDOW=168 horas, o modelo prevê as próximas
    HORIZON=24 horas de temperatura.
  - A janela avança de 1 em 1 hora (stride = 1).

Divisão temporal:
  - Treino:    01/01/2021 - 31/12/2023
  - Validação: 01/01/2024 - 31/12/2024
  - Teste:     01/01/2025 - 01/01/2026
  A divisão respeita a ordem temporal, evitando data leakage.

Referências:
  - Bengio, Y. et al. (2012). Practical recommendations for gradient-
    based training of deep architectures. In: Neural Networks: Tricks
    of the Trade, Springer.
  - PyTorch Documentation. torch.utils.data.Dataset.
    https://pytorch.org/docs/stable/data.html
"""

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Divisão temporal
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    train_end: str = "2024-01-01",    # 5Y
    val_end: str   = "2025-01-01",    # 5Y
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide o DataFrame em três partições cronológicas não sobrepostas.

    A separação temporal (em vez de aleatória) é obrigatória em
    séries temporais para evitar que informação futura vaze para o
    treino (leakage).

    Parâmetros
    ----------
    df        : pd.DataFrame  — deve conter coluna 'time' (datetime)
    train_end : str           — data limite (exclusive) do treino
    val_end   : str           — data limite (exclusive) da validação

    Retorna
    -------
    (train_df, val_df, test_df) : tuple de DataFrames
    """
    train_df = df[df["time"] < train_end].copy()
    val_df   = df[(df["time"] >= train_end) & (df["time"] < val_end)].copy()
    test_df  = df[df["time"] >= val_end].copy()

    logger.info(
        "Split temporal: treino=%d | validação=%d | teste=%d amostras",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Criação de janelas deslizantes
# ---------------------------------------------------------------------------

def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    window: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera sequências de entrada/saída por janela deslizante (stride=1).

    Para cada posição i, cria:
      - X_i = X[i : i+window]         → janela de contexto
      - y_i = y[i+window : i+window+horizon] → sequência-alvo futura

    Parâmetros
    ----------
    X      : np.ndarray, shape (T, n_features)  — features normalizadas
    y      : np.ndarray, shape (T,)             — target normalizado
    window : int  — número de passos no contexto (168 h)
    horizon: int  — número de passos a prever (24 h)

    Retorna
    -------
    (X_seq, y_seq)
      X_seq : shape (N, window, n_features)
      y_seq : shape (N, horizon)
    """
    n = len(X)
    # Número de janelas válidas
    n_seq = n - window - horizon + 1

    if n_seq <= 0:
        raise ValueError(
            f"Série temporal curta demais para window={window}, horizon={horizon}. "
            f"Mínimo necessário: {window + horizon} amostras, mas série tem {n}."
        )

    X_seqs = np.empty((n_seq, window, X.shape[1]), dtype=np.float32)
    y_seqs = np.empty((n_seq, horizon),             dtype=np.float32)

    for i in range(n_seq):
        X_seqs[i] = X[i : i + window]
        y_seqs[i] = y[i + window : i + window + horizon]

    logger.info("Janelas criadas: %d sequências de (%d, %d)", n_seq, window, X.shape[1])
    return X_seqs, y_seqs


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

class WeatherDataset(Dataset):
    """
    Dataset PyTorch para previsão meteorológica.

    Cada item é um par (X, y) onde:
      X : Tensor float32, shape (window, n_features) — janela de contexto
      y : Tensor float32, shape (horizon,)           — temperaturas futuras

    Parâmetros
    ----------
    X : np.ndarray, shape (N, window, n_features)
    y : np.ndarray, shape (N, horizon)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Conversão antecipada para Tensor economiza tempo durante o treinamento
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Construção dos DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    batch_size: int = 128,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Cria DataLoaders de treino, validação e teste.

    O conjunto de treino usa shuffle=True para quebrar correlações
    temporais locais entre batches, o que melhora a estabilidade do
    gradiente estocástico.  Validação e teste não são embaralhados para
    preservar a ordem cronológica durante a avaliação.

    Parâmetros
    ----------
    X_train, y_train : arrays de treino
    X_val,   y_val   : arrays de validação
    X_test,  y_test  : arrays de teste
    batch_size        : tamanho do mini-batch
    num_workers       : paralelismo de carregamento de dados

    Retorna
    -------
    (train_loader, val_loader, test_loader)
    """
    train_ds = WeatherDataset(X_train, y_train)
    val_ds   = WeatherDataset(X_val,   y_val)
    test_ds  = WeatherDataset(X_test,  y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(
        "DataLoaders criados: treino=%d | val=%d | teste=%d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )
    return train_loader, val_loader, test_loader
