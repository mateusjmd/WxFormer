"""
data/loader.py
==============
Etapas 1 e 2 do pipeline: leitura e merging dos arquivos NetCDF do ERA5-Land.

Referência:
  ECMWF. ERA5-Land hourly data from 1950 to present. Climate Data Store, 2019.
  DOI: 10.24381/cds.e2161bac
"""

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def load_and_merge_nc(file_paths: list[str]) -> pd.DataFrame:
    """
    Carrega e une múltiplos arquivos NetCDF do ERA5-Land em um único
    DataFrame ordenado cronologicamente.

    O ERA5-Land distribui variáveis em arquivos separados por tema
    (temperatura, pressão, radiação, solo, vento). Esta função usa
    ``xr.merge`` para combiná-los pelo eixo temporal compartilhado.

    Parâmetros
    ----------
    file_paths : list[str]
        Caminhos para os arquivos .nc do ERA5-Land.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas: valid_time, d2m, t2m, sp, tp, ssrd,
        swvl1, u10, v10.  Ordenado cronologicamente, sem índice temporal.

    Raises
    ------
    FileNotFoundError
        Se algum dos arquivos não for encontrado.
    """
    # Verifica existência de todos os arquivos antes de abrir
    for fp in file_paths:
        if not Path(fp).exists():
            raise FileNotFoundError(
                f"Arquivo NetCDF não encontrado: {fp}\n"
                "Certifique-se de que os arquivos estão em data/raw/"
            )

    logger.info("Abrindo %d arquivos NetCDF...", len(file_paths))
    datasets = [xr.open_dataset(fp) for fp in file_paths]

    # xr.merge combina datasets com dimensão temporal compartilhada
    # compat='override' suprime FutureWarning do xarray >= 2024
    logger.info("Realizando merge dos datasets...")
    ds = xr.merge(datasets, compat="override")

    # Converte para DataFrame tabular e reseta o índice para coluna
    df = ds.to_dataframe().reset_index()

    # Remove colunas de coordenadas geográficas fixas (ponto único)
    df = df.drop(columns=["latitude", "longitude"], errors="ignore")

    # Renomeia a dimensão temporal para nome canônico
    df = df.rename(columns={"valid_time": "time"})

    # Ordena cronologicamente e reseta índice inteiro
    df = df.sort_values("time").reset_index(drop=True)

    logger.info(
        "Dataset carregado: %d amostras, de %s a %s",
        len(df),
        df["time"].min(),
        df["time"].max(),
    )

    # Fecha os datasets para liberar recursos
    for ds_ in datasets:
        ds_.close()

    return df
