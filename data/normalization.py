"""
data/normalization.py
=====================
Etapa 4 do pipeline: normalização diferenciada por tipo de feature.

Estratégia de normalização:
  - Features físicas  → StandardScaler (média 0, desvio padrão 1)
  - Features cíclicas → sem normalização (já estão em [-1, 1])
  - Target (t2m)      → StandardScaler independente (necessário para
                        inverter a transformação nas previsões)

Os scalers são ajustados APENAS sobre o conjunto de treino e aplicados
a treino, validação e teste, prevenindo data leakage temporal.

Referências:
  - Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn,
    Keras, and TensorFlow (3ª ed.). O'Reilly Media. Cap. 2.
  - Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in
    Python. JMLR, 12, 2825-2830.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """
    Encapsula os scalers de features físicas e do target.

    Dois scalers são mantidos separados para permitir a inversão da
    normalização do target durante a avaliação — necessário para
    reportar métricas na escala original (°C).

    Atributos
    ---------
    feature_scaler : StandardScaler  — normaliza as features físicas
    target_scaler  : StandardScaler  — normaliza o target (t2m)
    physical_cols  : list[str]       — colunas normalizadas pelo feature_scaler
    cyclic_cols    : list[str]       — colunas passadas sem transformação
    target_col     : str             — nome do target
    """

    def __init__(
        self,
        physical_cols: list[str],
        cyclic_cols: list[str],
        target_col: str,
    ):
        self.physical_cols = physical_cols
        self.cyclic_cols   = cyclic_cols
        self.target_col    = target_col

        # StandardScaler para features físicas (média 0, std 1)
        self.feature_scaler = StandardScaler()

        # StandardScaler separado para o target — permite inversão
        self.target_scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Ajuste (fit) — apenas no conjunto de treino
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "FeatureNormalizer":
        """
        Calcula média e desvio padrão das features físicas e do target
        usando SOMENTE os dados de treino.

        Parâmetros
        ----------
        train_df : pd.DataFrame  — partição de treino

        Retorna
        -------
        self  (para encadeamento)
        """
        self.feature_scaler.fit(train_df[self.physical_cols])
        # Target precisa de shape (N,1) para o scaler univariado
        self.target_scaler.fit(train_df[[self.target_col]])
        logger.info("Scalers ajustados sobre o conjunto de treino.")
        return self

    # ------------------------------------------------------------------
    # Transformação
    # ------------------------------------------------------------------

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Aplica normalização e retorna a matriz de features.

        Features físicas são normalizadas (StandardScaler).
        Features cíclicas são concatenadas sem transformação.

        Parâmetros
        ----------
        df : pd.DataFrame

        Retorna
        -------
        np.ndarray, shape (N, n_features)
        """
        # Normaliza features físicas
        phys_norm = self.feature_scaler.transform(df[self.physical_cols])

        # Features cíclicas: mantidas em [-1, 1] como estão
        cyc = df[self.cyclic_cols].values

        # Concatenação na ordem: físicas || cíclicas
        return np.concatenate([phys_norm, cyc], axis=1)

    def transform_target(self, df: pd.DataFrame) -> np.ndarray:
        """
        Normaliza o target (t2m).

        Parâmetros
        ----------
        df : pd.DataFrame

        Retorna
        -------
        np.ndarray, shape (N,)
        """
        return self.target_scaler.transform(df[[self.target_col]]).ravel()

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Reverte a normalização do target, retornando valores em °C.

        Parâmetros
        ----------
        y : np.ndarray  — previsões normalizadas, shape (N,) ou (N, H)

        Retorna
        -------
        np.ndarray  — valores em °C, mesma shape que y
        """
        original_shape = y.shape
        y_flat = y.reshape(-1, 1)
        y_inv  = self.target_scaler.inverse_transform(y_flat)
        return y_inv.reshape(original_shape)

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Salva os scalers em disco (pickle)."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("FeatureNormalizer salvo em %s", path)

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        """Carrega scalers do disco."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("FeatureNormalizer carregado de %s", path)
        return obj
