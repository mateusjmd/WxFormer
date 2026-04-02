"""
data/features.py
================
Etapa 3 do pipeline: engenharia de features físicas e temporais.

Operações realizadas:
  a) Conversões físicas de unidades (Kelvin → Celsius, m → mm, etc.)
  b) Derivação de wind_speed a partir de u10 e v10
  c) Codificação cíclica de hora do dia e dia do ano (sin/cos)

Referências:
  - Bechtold, P. et al. (2014). Atmospheric moisture convection.
    ECMWF Lecture Notes.
  - Rasp, S. et al. (2020). WeatherBench: A benchmark for data-driven
    weather forecasting. JAMES, 12(11). DOI: 10.1029/2020MS002203
  - Cleveland, W. S., & Devlin, S. J. (1988). Locally weighted
    regression. JASA, 83(403), 596–610.  (codificação cíclica)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversões físicas
# ---------------------------------------------------------------------------

def convert_temperatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte temperatura do ar (t2m) e temperatura de ponto de orvalho
    (d2m) de Kelvin para graus Celsius.

    ERA5-Land fornece temperaturas em Kelvin (unidade SI padrão do ECMWF).
    A conversão linear T_C = T_K - 273.15 é aplicada a ambas as variáveis.

    Parâmetros
    ----------
    df : pd.DataFrame  (modifica in-place e retorna)

    Retorna
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["t2m"] = df["t2m"] - 273.15  # K → °C
    df["d2m"] = df["d2m"] - 273.15  # K → °C
    logger.debug("Temperaturas convertidas de K para °C.")
    return df


def convert_precipitation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte precipitação total (tp) de metros acumulados por hora para
    milímetros por hora.

    ERA5-Land armazena tp em metros de lâmina d'água equivalente.
    Multiplicação por 1000 converte para mm, que é a unidade operacional
    padrão em meteorologia.

    Parâmetros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["tp"] = df["tp"] * 1000.0  # m → mm
    logger.debug("Precipitação convertida de m para mm.")
    return df


def add_wind_speed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a velocidade escalar do vento a 10 m (wind_speed) a partir
    das componentes zonal (u10) e meridional (v10).

        wind_speed = sqrt(u10² + v10²)

    A norma euclidiana preserva toda a informação energética das
    componentes vetoriais e é amplamente usada como feature meteorológica
    em modelos de ML (Rasp et al., 2020).

    Parâmetros
    ----------
    df : pd.DataFrame  — deve conter as colunas 'u10' e 'v10'

    Retorna
    -------
    pd.DataFrame  — com coluna adicional 'wind_speed' (m/s)
    """
    df = df.copy()
    df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    logger.debug("Feature 'wind_speed' calculada.")
    return df


# ---------------------------------------------------------------------------
# Codificação cíclica de features temporais
# ---------------------------------------------------------------------------

def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona representações cíclicas do horário (hora do dia) e da
    sazonalidade (dia do ano) usando pares sin/cos.

    A codificação cíclica f(t) = (sin(2π·t/T), cos(2π·t/T)) mapeia
    valores periódicos para um espaço 2-D contínuo onde a distância
    euclidiana reflete a proximidade temporal real (Cleveland & Devlin,
    1988).  Isso evita descontinuidades artificiais que surgem ao tratar
    hora=0 e hora=23 como distantes — o que ocorreria com representação
    inteira ou one-hot.

    Features criadas:
      hour_sin, hour_cos  — ciclo diário de 24 h
      doy_sin,  doy_cos   — ciclo sazonal de 365 dias

    Parâmetros
    ----------
    df : pd.DataFrame  — deve conter coluna 'time' (datetime)

    Retorna
    -------
    pd.DataFrame
    """
    df = df.copy()

    hour = df["time"].dt.hour
    day_of_year = df["time"].dt.dayofyear

    # Ciclo diário (período T = 24 horas)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Ciclo sazonal (período T = 365 dias)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    logger.debug("Features cíclicas (hour_sin/cos, doy_sin/cos) adicionadas.")
    return df


# ---------------------------------------------------------------------------
# Pipeline completo de feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas as transformações de feature engineering em sequência:
      1. Conversão de temperatura (K → °C)
      2. Conversão de precipitação (m → mm)
      3. Derivação de wind_speed
      4. Codificação cíclica hora/sazonalidade

    Parâmetros
    ----------
    df : pd.DataFrame  — DataFrame bruto do loader (coluna 'time' obrigatória)

    Retorna
    -------
    pd.DataFrame  — DataFrame enriquecido com todas as features
    """
    logger.info("Iniciando feature engineering...")
    df = convert_temperatures(df)
    df = convert_precipitation(df)
    df = add_wind_speed(df)
    df = add_cyclic_time_features(df)
    logger.info(
        "Feature engineering concluído. Colunas: %s", list(df.columns)
    )
    return df
