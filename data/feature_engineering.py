"""
data/feature_engineering.py
============================
Passo 3 do pipeline: Engenharia de features físicas.

Realiza:
 1. Conversão de unidades (K → °C, m → mm, etc.)
 2. Codificação cíclica de variáveis temporais (hora, dia do ano)
 3. Cálculo do déficit de pressão de vapor (VPD) — proxy de estresse hídrico
 4. Cálculo da velocidade escalar do vento a partir dos componentes U/V

Justificativa das features
--------------------------
- Codificação cíclica com sin/cos preserva a continuidade circular
  do tempo (ex.: hora 23 é próxima da hora 0) sem introduzir
  artefatos de ordenação [1].
- O VPD é reconhecido como preditor importante de temperatura de
  superfície e demanda evapotranspirativa [2].
- A velocidade do vento escalar é fisicamente mais interpretável para
  modelos do que os componentes U/V separados, especialmente quando
  a direção não é alvo da previsão [3].

Referências
-----------
[1] Vaswani, A. et al. Attention Is All You Need. NeurIPS 2017.
[2] Allen, R. G. et al. FAO Irrigation and Drainage Paper No. 56.
    FAO, Rome, 1998.
[3] Hersbach, H. et al. The ERA5 global reanalysis. Q. J. R.
    Meteorol. Soc., 146, 1999–2049, 2020. DOI: 10.1002/qj.3803
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Constantes físicas
# ─────────────────────────────────────────────
KELVIN_OFFSET = 273.15   # Deslocamento Kelvin → Celsius
MM_PER_METER  = 1000.0   # Fator de conversão m → mm


def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte variáveis ERA5-Land das unidades nativas do CDS para
    unidades mais intuitivas usadas na literatura meteorológica.

    Conversões realizadas
    ---------------------
    - t2m  : K  → °C  (temperatura do ar a 2 m)
    - d2m  : K  → °C  (temperatura do ponto de orvalho a 2 m)
    - tp   : m  → mm  (precipitação total acumulada por hora)

    As demais variáveis (sp em Pa, ssrd em J/m², swvl1 em m³/m³,
    u10/v10 em m/s) já estão em unidades padronizadas e não requerem
    conversão.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com variáveis ERA5-Land nas unidades nativas.

    Retorna
    -------
    pd.DataFrame
        DataFrame com unidades convertidas (operação in-place sobre
        uma cópia para preservar o original).
    """
    df = df.copy()

    # Temperatura do ar e do ponto de orvalho: Kelvin → Celsius
    for col in ["t2m", "d2m"]:
        if col in df.columns:
            df[col] = df[col] - KELVIN_OFFSET
            logger.debug("Convertido %s: K → °C", col)

    # Precipitação: metros acumulados por hora → milímetros
    if "tp" in df.columns:
        df["tp"] = df["tp"] * MM_PER_METER
        # Clipa valores negativos (artefatos numéricos do modelo)
        df["tp"] = df["tp"].clip(lower=0.0)
        logger.debug("Convertido tp: m → mm (valores negativos clipados)")

    return df


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona codificações cíclicas para variáveis temporais periódicas.

    A codificação sin/cos é a representação padrão para variáveis
    circulares em aprendizado de máquina, pois:
      - Mantém a continuidade na fronteira do ciclo (ex.: 23h→0h).
      - Preserva a distância angular natural entre instantes [1].

    Features criadas
    ----------------
    - hour_sin, hour_cos : ciclo diário (período = 24 h)
    - doy_sin, doy_cos   : ciclo anual (período = 365 dias)

    As colunas auxiliares 'hour' e 'day_of_year' são removidas após
    a codificação.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com coluna 'time' do tipo datetime.

    Retorna
    -------
    pd.DataFrame
        DataFrame com as quatro novas features cíclicas.
    """
    df = df.copy()

    # Extrai hora e dia do ano a partir do índice temporal
    df["hour"]        = df["time"].dt.hour
    df["day_of_year"] = df["time"].dt.dayofyear

    # Codificação cíclica: sin e cos do ângulo normalizado [0, 2π]
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]  = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Remove colunas auxiliares brutas (não utilizadas pelo modelo)
    df = df.drop(columns=["hour", "day_of_year"])

    logger.debug("Features cíclicas criadas: hour_sin, hour_cos, doy_sin, doy_cos")
    return df


def add_vpd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula e adiciona o Déficit de Pressão de Vapor (VPD) em hPa.

    O VPD mede a diferença entre a pressão de vapor de saturação
    e a pressão de vapor atual, sendo um indicador de demanda
    evapotranspirativa e estresse térmico [2].

    Fórmula de Tetens (aproximação de Magnus):
        e_s(T) = 6.112 × exp(17.67 × T / (T + 243.5))   [hPa]
        VPD    = e_s(t2m) - e_s(d2m)

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com colunas 't2m' e 'd2m' já em °C.

    Retorna
    -------
    pd.DataFrame
        DataFrame com coluna adicional 'vpd' (hPa).
    """
    df = df.copy()

    if "t2m" not in df.columns or "d2m" not in df.columns:
        logger.warning("Colunas 't2m' ou 'd2m' ausentes — VPD não calculado.")
        return df

    def magnus(T: pd.Series) -> pd.Series:
        """Pressão de vapor de saturação pela fórmula de Magnus [hPa]."""
        return 6.112 * np.exp(17.67 * T / (T + 243.5))

    e_sat  = magnus(df["t2m"])   # pressão de saturação na temperatura do ar
    e_dew  = magnus(df["d2m"])   # pressão de vapor atual (≈ saturação no orvalho)
    df["vpd"] = (e_sat - e_dew).clip(lower=0.0)

    logger.debug("Feature VPD calculada (fórmula de Magnus).")
    return df


def add_wind_speed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a velocidade escalar do vento (módulo do vetor horizontal)
    a partir dos componentes zonal (u10) e meridional (v10).

    wind_speed = sqrt(u10² + v10²)

    A velocidade escalar é uma feature mais estável e interpretável
    que os componentes individuais para modelos de previsão de
    temperatura, que não precisam de informação direcional do vento [3].

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com colunas 'u10' e 'v10' (m/s).

    Retorna
    -------
    pd.DataFrame
        DataFrame com coluna adicional 'wind_speed' (m/s).
    """
    df = df.copy()

    if "u10" not in df.columns or "v10" not in df.columns:
        logger.warning("Colunas 'u10' ou 'v10' ausentes — wind_speed não calculado.")
        return df

    df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    logger.debug("Feature wind_speed calculada.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa toda a engenharia de features em sequência:
      1. Conversão de unidades
      2. Codificação cíclica temporal
      3. Cálculo do VPD
      4. Velocidade escalar do vento

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame bruto após load_and_merge().

    Retorna
    -------
    pd.DataFrame
        DataFrame enriquecido com todas as features físicas.
    """
    logger.info("Iniciando engenharia de features físicas ...")

    df = convert_units(df)
    df = add_cyclic_features(df)
    df = add_vpd(df)
    df = add_wind_speed(df)

    logger.info(
        "Engenharia de features concluída. Shape: %s", df.shape
    )
    return df
