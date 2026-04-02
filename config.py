"""
config.py
=========
Configuração central do projeto Weather Transformer.

Todas as constantes, caminhos e hiperparâmetros padrão estão
centralizados aqui para facilitar a manutenção e reprodutibilidade.

Diretórios com timestamp
------------------------
Cada execução cria automaticamente subdiretórios nomeados com o
horário de início (formato YYYY-MM-DD_HH-MM-SS) dentro de
`checkpoints/` e `outputs/`. Isso isola completamente os artefatos
de execuções distintas, permitindo comparação direta de resultados
sem risco de sobrescrita.

    checkpoints/
        2026-03-10_14-05-30/
            best_model.pt
            normalizer.pkl
    outputs/
        2026-03-10_14-05-30/
            run.log
            learning_curves.png
            predictions.png
            attention_heatmap.png
            saliency_heatmap.png
            optuna.db
"""

import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Caminhos base (invariantes entre execuções)
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

# ---------------------------------------------------------------------------
# Timestamp da execução atual
# ---------------------------------------------------------------------------

# Gerado uma única vez no momento da importação do módulo.
# Todos os módulos que importam `config` recebem o mesmo timestamp,
# garantindo consistência entre checkpoints, logs e figuras.
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ---------------------------------------------------------------------------
# Diretórios com timestamp (criados automaticamente)
# ---------------------------------------------------------------------------

# Diretório raiz de checkpoints e outputs (contém todas as execuções)
_CHECKPOINTS_ROOT = os.path.join(BASE_DIR, "checkpoints")
_OUTPUTS_ROOT     = os.path.join(BASE_DIR, "outputs")

# Subdiretório exclusivo desta execução
CHECKPOINT_DIR = os.path.join(_CHECKPOINTS_ROOT, RUN_TIMESTAMP)
OUTPUT_DIR     = os.path.join(_OUTPUTS_ROOT,     RUN_TIMESTAMP)

# Cria os diretórios se ainda não existirem
for _dir in [CHECKPOINT_DIR, OUTPUT_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Arquivos NetCDF do ERA5-Land
# ---------------------------------------------------------------------------

# 5Y
NC_FILES = [
    os.path.join(DATA_DIR, "5Y/reanalysis-era5-land-timeseries-sfc-2m-temperature5akfe65q.nc"),
    os.path.join(DATA_DIR, "5Y/reanalysis-era5-land-timeseries-sfc-pressure-precipitationtnmndk_6.nc"),
    os.path.join(DATA_DIR, "5Y/reanalysis-era5-land-timeseries-sfc-radiation-heatse04ic7b.nc"),
    os.path.join(DATA_DIR, "5Y/reanalysis-era5-land-timeseries-sfc-soil-water69miog7u.nc"),
    os.path.join(DATA_DIR, "5Y/reanalysis-era5-land-timeseries-sfc-windzqfeh_51.nc"),
]

# 10 Y
#NC_FILES = [
#    os.path.join(DATA_DIR, "10Y/reanalysis-era5-land-timeseries-sfc-2m-temperature15m49hx2.nc"),
#    os.path.join(DATA_DIR, "10Y/reanalysis-era5-land-timeseries-sfc-pressure-precipitationyxqiyesm.nc"),
#    os.path.join(DATA_DIR, "10Y/reanalysis-era5-land-timeseries-sfc-radiation-heatlimu8lx2.nc"),
#    os.path.join(DATA_DIR, "10Y/reanalysis-era5-land-timeseries-sfc-soil-watersva5mxu3.nc"),
#    os.path.join(DATA_DIR, "10Y/reanalysis-era5-land-timeseries-sfc-wind4yos7ijw.nc")
#]

# 50 Y
#NC_FILES = [
#    os.path.join(DATA_DIR, "50Y/reanalysis-era5-land-timeseries-sfc-2m-temperature3mx3yb5v.nc"),
#    os.path.join(DATA_DIR, "50Y/reanalysis-era5-land-timeseries-sfc-pressure-precipitation5s6w8b8u.nc"),
#    os.path.join(DATA_DIR, "50Y/reanalysis-era5-land-timeseries-sfc-radiation-heatb2b69eai.nc"),
#    os.path.join(DATA_DIR, "50Y/reanalysis-era5-land-timeseries-sfc-soil-waterxmkry1bx.nc"),
#    os.path.join(DATA_DIR, "50Y/reanalysis-era5-land-timeseries-sfc-windlts9u68d.nc")
#]

# 75 Y
#NC_FILES = [
#    os.path.join(DATA_DIR, "75Y/reanalysis-era5-land-timeseries-sfc-2m-temperatureebhfrd8s.nc"),
#    os.path.join(DATA_DIR, "75Y/reanalysis-era5-land-timeseries-sfc-pressure-precipitationk2bvs8ql.nc"),
#    os.path.join(DATA_DIR, "75Y/reanalysis-era5-land-timeseries-sfc-radiation-heatrud3moo8.nc"),
#    os.path.join(DATA_DIR, "75Y/reanalysis-era5-land-timeseries-sfc-soil-waterx4pm2q96.nc"),
#    os.path.join(DATA_DIR, "75Y/reanalysis-era5-land-timeseries-sfc-windtlugg534.nc")
#]

# ---------------------------------------------------------------------------
# Variável-alvo e features
# ---------------------------------------------------------------------------

TARGET = "t2m"

# Features físicas — recebem normalização via StandardScaler
PHYSICAL_FEATURES = ["d2m", "sp", "tp", "ssrd", "swvl1", "u10", "v10", "wind_speed"]

# Features cíclicas — em [-1,1] por construção; NÃO normalizadas
CYCLIC_FEATURES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

FEATURES  = PHYSICAL_FEATURES + CYCLIC_FEATURES  # 12 features
INPUT_DIM = len(FEATURES)                         # 12

# ---------------------------------------------------------------------------
# Localização geográfica
# ---------------------------------------------------------------------------

LATITUDE  = -22.9
LONGITUDE = -47.1

# ---------------------------------------------------------------------------
# Divisão temporal
# ---------------------------------------------------------------------------

TRAIN_END = "2024-01-01" # 5Y
VAL_END   = "2025-01-01" # 5Y
#TRAIN_END = "2023-01-01"  # 10 Y
#VAL_END = "2024-01-01"    # 10Y
#TRAIN_END = "2011-01-01" # 50Y
#VAL_END = "2019-01-01"   # 50Y
#TRAIN_END = "2001-01-01"  # 75Y
#VAL_END = "2013-01-01"    # 75Y

# ---------------------------------------------------------------------------
# Janelas temporais
# ---------------------------------------------------------------------------

WINDOW     = 168  # 1 semana de contexto
HORIZON    = 24   # 24 h à frente
PATCH_SIZE = 6    # 6 h por patch → 28 patches

# ---------------------------------------------------------------------------
# Hiperparâmetros padrão para treinamento final
# ---------------------------------------------------------------------------

DEFAULT_EPOCHS       = 100
DEFAULT_PATIENCE     = 15
DEFAULT_BATCH_SIZE   = 128
DEFAULT_LR           = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
GRAD_CLIP_MAX_NORM   = 1.0

# ---------------------------------------------------------------------------
# Configurações do Optuna
# ---------------------------------------------------------------------------

N_TRIALS        = 100
OPTUNA_EPOCHS   = 50
OPTUNA_PATIENCE = 10

SEED = 42
