import logging
from pathlib import Path

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

def load_and_merge_nc(file_paths: list[str]) -> pd.DataFrame:

    # Verifica a existência de todos os arquivos antes de abrir
    for fp in file_paths:
        if not Path(fp).exists():
            raise FileNotFoundError(
            f"Arquivo NetCDF não encontrado: {fp}\n"
            "Certifique-se de que os arquivos estão em data/raw/"
            )
        
    logger.info("Abrindo %d arquivos NetCDF...", len(file_paths))
    datasets = [xr.open_dataset(fp) for fp in file_paths]

    logger.info("Realizando merge dos datasets...")
    ds = xr.merge(datasets, compat='override')

    # converte para DataFrame tabular e reseta o índice para coluna
    df = ds.to_dataframe().reset_index()

    # Remove colunas de coordenadas geográficas fixas (ponto único)
    df = df.drop(columns=['latitude', 'longitude'], errors='ignore')

    # Renomeia a dimensão temporal para um nome canônico
    df = df.rename(columns={'valid_time': 'time'})

    # Ordena cronologicamente e resta índice inteiro
    df = df.sort_values('time').reset_index(drop=True)

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