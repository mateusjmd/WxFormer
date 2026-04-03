"""
inference.py
============
Inferência pontual do WxFormer sobre um dia específico de dados ERA5-Land
não vistos durante nenhuma fase do treinamento.

Compara hora a hora as previsões com os valores reais de t2m e reporta
MSE, RMSE e MAE na escala original (°C).

Posicionamento
--------------
Execute a partir do diretório raiz do projeto (mesmo nível de main.py):

    WxFormer/
    ├── config.py
    ├── main.py
    ├── inference.py        ← aqui
    ├── data/
    ├── model/
    └── ...

Arquitetura dos dados na inferência
------------------------------------
O script opera com duas fontes distintas de dados:

  1. cfg.NC_FILES (arquivos de treino)
     Contêm todas as variáveis ERA5-Land até 01-01-2026. Deste conjunto
     são extraídas as 168 horas de contexto (31-12-2025 às 01-01-2026 23:00)
     necessárias para alimentar o modelo. As features são construídas com
     build_features() e normalizadas com o FeatureNormalizer do treino.

  2. Arquivo NC separado — apenas t2m (ground truth do dia alvo)
     Contém somente a variável t2m para 02-01-2026 (00:00 → 23:00).
     É usado exclusivamente como alvo real para comparação; NÃO é usado
     como entrada do modelo. A conversão K → °C é aplicada manualmente
     (sem build_features, pois as demais variáveis não estão presentes).

Hiperparâmetros
---------------
Os hiperparâmetros do melhor trial são extraídos diretamente do banco
SQLite gerado pelo Optuna (optuna.db), sem necessidade de re-treino ou
de arquivos JSON adicionais.

Uso
---
    python inference.py \\
        --checkpoint  checkpoints/2026-01-01_10-00-00/best_model.pt  \\
        --normalizer  checkpoints/2026-01-01_10-00-00/normalizer.pkl \\
        --optuna_db   outputs/2026-01-01_10-00-00/optuna.db          \\
        --gt_nc       data/raw/reanalysis-era5-land-<hash>-t2m-20260102.nc \\
        --target_date 2026-01-02

Argumentos opcionais:
    --study_name  nome do estudo Optuna (default: "weather_transformer")
    --output_dir  diretório de saída    (default: inference_results/)
    --device      auto | cpu | cuda | mps (default: auto)

Saídas
------
  inference_results/
      inference_<data>.pdf   — gráfico hora a hora (real vs. previsto + erro)
      metrics_<data>.txt     — métricas globais e tabela hora a hora
"""

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import optuna

# ---------------------------------------------------------------------------
# Módulos internos do projeto
# ---------------------------------------------------------------------------
import config as cfg
from data.loader        import load_and_merge_nc
from data.features      import build_features
from data.normalization import FeatureNormalizer
from model.transformer  import WeatherTransformer
from training.evaluate  import _apply_paper_style, compute_metrics

# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt  = "%H:%M:%S",
    handlers = [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("inference")

# Silencia os logs verbosos do Optuna durante o carregamento do estudo
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# 1. Extração de hiperparâmetros do banco Optuna
# ============================================================================

def load_hparams_from_db(db_path: Path, study_name: str) -> dict:
    """
    Carrega os hiperparâmetros do melhor trial diretamente do banco SQLite
    gerado pelo Optuna durante o treinamento.

    Aplica o mesmo pós-processamento de best_hyperparams_from_study() em
    tuning/optuna_search.py: recalcula dim_ff a partir de ff_mult × d_model.

    Parâmetros
    ----------
    db_path    : Path — caminho para o arquivo optuna.db
    study_name : str  — nome do estudo (default: "weather_transformer")

    Retorna
    -------
    dict com os hiperparâmetros prontos para instanciar WeatherTransformer
    """
    storage = f"sqlite:///{db_path}"
    study   = optuna.load_study(study_name=study_name, storage=storage)

    best    = study.best_trial
    params  = best.params.copy()

    # Recalcula dim_ff — mesmo cálculo de best_hyperparams_from_study()
    if "ff_mult" in params:
        params["dim_ff"] = params.pop("ff_mult") * params["d_model"]

    log.info(
        "Melhor trial #%d carregado do Optuna | val_loss = %.6f",
        best.number, best.value,
    )
    log.info("Hiperparâmetros: %s", params)
    return params


# ============================================================================
# 2. Carregamento do modelo
# ============================================================================

def load_model(
    checkpoint_path: Path,
    hparams:         dict,
    device:          torch.device,
) -> torch.nn.Module:
    """
    Reconstrói o WeatherTransformer com os hiperparâmetros do Optuna e
    carrega o state_dict salvo pelo EarlyStopping.

    Parâmetros
    ----------
    checkpoint_path : Path         — caminho para best_model.pt
    hparams         : dict         — hiperparâmetros extraídos do Optuna
    device          : torch.device

    Retorna
    -------
    WeatherTransformer em modo eval
    """
    model = WeatherTransformer(
        input_dim    = cfg.INPUT_DIM,
        d_model      = hparams.get("d_model",      256),
        n_heads      = hparams.get("n_heads",        4),
        n_layers     = hparams.get("n_layers",       4),
        dim_ff       = hparams.get("dim_ff",       1024),
        attn_dropout = hparams.get("attn_dropout",  0.1),
        ff_dropout   = hparams.get("ff_dropout",    0.1),
        horizon      = cfg.HORIZON,
        seq_len      = cfg.WINDOW,
        patch_size   = cfg.PATCH_SIZE,
        use_alibi    = True,
    ).to(device)

    state_dict = torch.load(
        checkpoint_path,
        map_location = device,
        weights_only = True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    log.info(
        "Modelo carregado de %s | parâmetros treináveis: %d",
        checkpoint_path, model.count_parameters(),
    )
    return model


# ============================================================================
# 3. Ground truth: carrega t2m do arquivo NC separado
# ============================================================================

def load_ground_truth(gt_nc_path: Path, target_date: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Lê a temperatura real a 2 m do arquivo NetCDF do dia alvo.

    O arquivo contém apenas a variável t2m (em Kelvin, padrão ERA5-Land)
    referente ao dia alvo. A conversão K → °C é aplicada diretamente,
    sem chamar build_features() (as demais variáveis não estão presentes).

    Parâmetros
    ----------
    gt_nc_path  : Path — caminho para o .nc com t2m do dia alvo
    target_date : str  — "YYYY-MM-DD" (usado para validação dos timestamps)

    Retorna
    -------
    y_true     : np.ndarray, shape (HORIZON,) — temperaturas reais em °C
    timestamps : pd.DatetimeIndex             — 24 timestamps do dia alvo
    """
    ds = xr.open_dataset(gt_nc_path)

    # Identifica a dimensão temporal (ERA5-Land usa 'valid_time' ou 'time')
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"

    # Converte para DataFrame e padroniza o nome da coluna temporal
    df = ds[[cfg.TARGET]].to_dataframe().reset_index()
    df = df.rename(columns={time_dim: "time"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    ds.close()

    # Filtra apenas o dia alvo (00:00 → 23:00)
    target_dt = pd.Timestamp(target_date)
    pred_end  = target_dt + timedelta(hours=cfg.HORIZON - 1)
    mask      = (df["time"] >= target_dt) & (df["time"] <= pred_end)
    day_df    = df[mask].reset_index(drop=True)

    if len(day_df) != cfg.HORIZON:
        raise ValueError(
            f"Arquivo de ground truth incompleto: esperados {cfg.HORIZON} "
            f"registros para {target_date}, obtidos {len(day_df)}. "
            f"Verifique o arquivo {gt_nc_path}."
        )

    # K → °C (mesmo offset aplicado por convert_temperatures() em features.py)
    y_true     = (day_df[cfg.TARGET].values - 273.15).astype(np.float32)
    timestamps = pd.DatetimeIndex(day_df["time"].values)

    log.info(
        "Ground truth carregado de %s | t2m: min=%.2f°C max=%.2f°C",
        gt_nc_path, y_true.min(), y_true.max(),
    )
    return y_true, timestamps


# ============================================================================
# 4. Janela de contexto (168 h de features dos arquivos de treino)
# ============================================================================

def build_context_window(
    df:         pd.DataFrame,
    normalizer: FeatureNormalizer,
    target_date: str,
) -> torch.Tensor:
    """
    Extrai a janela de contexto de cfg.WINDOW horas imediatamente anteriores
    ao dia alvo, normaliza com o FeatureNormalizer do treino e empacota
    no tensor de entrada do modelo.

    O DataFrame deve ter sido processado por build_features() (todas as
    12 features presentes, t2m em °C). A coluna 'time' é convertida para
    índice DatetimeIndex internamente para slicing eficiente.

    A janela vai de (target_date - 168h) até (target_date - 1h), inclusive,
    replicando exatamente a lógica da janela deslizante do pipeline de treino.

    Parâmetros
    ----------
    df          : pd.DataFrame — DataFrame pós build_features(), coluna 'time'
    normalizer  : FeatureNormalizer — carregado de normalizer.pkl
    target_date : str — "YYYY-MM-DD"

    Retorna
    -------
    torch.Tensor, shape (1, WINDOW, n_features)
    """
    target_dt = pd.Timestamp(target_date)
    ctx_start = target_dt - timedelta(hours=cfg.WINDOW)  # 31-12-2025 00:00
    ctx_end   = target_dt - timedelta(hours=1)           # 01-01-2026 23:00

    # Indexa por 'time' para slicing por data
    df_idx = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_idx["time"]):
        df_idx["time"] = pd.to_datetime(df_idx["time"])
    df_idx = df_idx.set_index("time").sort_index()

    # Valida limites temporais
    if ctx_start < df_idx.index[0]:
        raise ValueError(
            f"Janela de contexto começa em {ctx_start}, mas os dados de treino "
            f"só existem a partir de {df_idx.index[0]}."
        )
    if ctx_end > df_idx.index[-1]:
        raise ValueError(
            f"Janela de contexto termina em {ctx_end}, mas os dados de treino "
            f"só existem até {df_idx.index[-1]}. Verifique cfg.NC_FILES."
        )

    ctx_df = df_idx.loc[ctx_start:ctx_end]

    if len(ctx_df) != cfg.WINDOW:
        raise ValueError(
            f"Janela de contexto incompleta: esperados {cfg.WINDOW} registros, "
            f"obtidos {len(ctx_df)}. Pode haver horas faltantes no NetCDF."
        )

    log.info(
        "Contexto: %s → %s (%d h)",
        ctx_start, ctx_end, len(ctx_df),
    )

    # Normaliza apenas as features (sem re-fit)
    X_norm   = normalizer.transform_features(ctx_df)           # (WINDOW, 12)
    x_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0)  # (1, W, 12)
    return x_tensor


# ============================================================================
# 5. Inferência e desnormalização
# ============================================================================

@torch.no_grad()
def run_inference(
    model:      torch.nn.Module,
    x_tensor:   torch.Tensor,
    normalizer: FeatureNormalizer,
    device:     torch.device,
) -> np.ndarray:
    """
    Executa o forward pass e reverte a normalização para °C.

    Parâmetros
    ----------
    model      : WeatherTransformer em modo eval
    x_tensor   : torch.Tensor, shape (1, WINDOW, n_features)
    normalizer : FeatureNormalizer (contém target_scaler do treino)
    device     : torch.device

    Retorna
    -------
    np.ndarray, shape (HORIZON,) — previsões em °C
    """
    y_norm    = model(x_tensor.to(device))               # (1, HORIZON)
    y_norm_np = y_norm.squeeze(0).cpu().numpy()          # (HORIZON,)
    y_pred    = normalizer.inverse_transform_target(y_norm_np)
    return y_pred.astype(np.float32)


# ============================================================================
# 6. Visualização — dois PDFs separados
# ============================================================================

def plot_temperature(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    target_date: str,
    output_path: Path,
) -> None:
    """
    PDF 1 — 2 m air temperature: ERA5-Land observed vs. WxFormer forecast.

    Layout: IEEE single-column (3.5 x 2.8 in).
    Legend placed below the plot area (outside axes) to avoid any overlap
    with the data curves. Metrics annotated in the lower-right dead zone,
    below the minimum temperature region where both curves converge.

    Parameters
    ----------
    y_true      : observed temperatures in degrees C
    y_pred      : WxFormer forecast in degrees C
    target_date : "YYYY-MM-DD" (used in the axis title)
    output_path : output file path (.pdf)
    """
    hours  = np.arange(cfg.HORIZON)
    C_BLUE = "#4477AA"
    C_RED  = "#EE6677"

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.subplots_adjust(bottom=0.30, top=0.88, left=0.14, right=0.97)

    ax.plot(hours, y_true, color=C_BLUE, lw=1.4,
            label="ERA5-Land (observed)")
    ax.plot(hours, y_pred, color=C_RED,  lw=1.4, ls="--",
            label="WxFormer (forecast)")
    ax.fill_between(hours, y_true, y_pred, alpha=0.13, color=C_RED)

    ax.set_title(f"Point Inference — {target_date}", pad=5, fontsize=10)
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("2 m air temperature (\u00b0C)")
    ax.set_xticks(hours[::3])
    ax.set_xticklabels([f"{h:02d}:00" for h in hours[::3]],
                       rotation=45, ha="right")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend outside the axes, centred below the x-axis label
    ax.legend(
        loc            = "upper center",
        bbox_to_anchor = (0.5, -0.38),
        ncol           = 2,
        framealpha     = 0.0,
        edgecolor      = "none",
    )

    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info("Temperature plot saved to %s", output_path)


def plot_hourly_error(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    metrics:     dict,
    target_date: str,
    output_path: Path,
) -> None:
    """
    PDF 2 — hourly forecast error (WxFormer minus ERA5-Land) with global
    metrics annotated inside the plot.

    Layout: IEEE single-column (3.5 x 2.2 in).
    Metrics box is placed in the upper-left corner, which is reliably
    empty because positive errors dominate the afternoon hours (right side)
    while early hours show near-zero error.

    Parameters
    ----------
    y_true      : observed temperatures in degrees C
    y_pred      : WxFormer forecast in degrees C
    metrics     : dict with keys mse, rmse, mae
    target_date : "YYYY-MM-DD"
    output_path : output file path (.pdf)
    """
    hours  = np.arange(cfg.HORIZON)
    error  = y_pred - y_true
    C_BLUE = "#4477AA"
    C_RED  = "#EE6677"

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    fig.subplots_adjust(bottom=0.22, top=0.88, left=0.14, right=0.97)

    bar_colors = [C_RED if e > 0 else C_BLUE for e in error]
    ax.bar(hours, error, color=bar_colors, width=0.72, alpha=0.85, zorder=3)
    ax.axhline(0, color="0.35", lw=0.7, zorder=4)

    ax.set_title(f"Hourly Forecast Error — {target_date}", pad=5, fontsize=10)
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Error (\u00b0C)")
    ax.set_xticks(hours[::3])
    ax.set_xticklabels([f"{h:02d}:00" for h in hours[::3]],
                       rotation=45, ha="right")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Metrics box anchored to axes fraction — upper left, reliably clear of bars
    metric_str = (
        f"MSE  = {metrics['mse']:.4f} \u00b0C\u00b2\n"
        f"RMSE = {metrics['rmse']:.4f} \u00b0C\n"
        f"MAE  = {metrics['mae']:.4f} \u00b0C"
    )
    ax.text(
        0.02, 0.97, metric_str,
        transform  = ax.transAxes,
        ha         = "left",
        va         = "top",
        fontsize   = 7,
        fontfamily = "monospace",
        bbox       = dict(boxstyle="round,pad=0.35", fc="white",
                          ec="0.72", lw=0.6, alpha=0.9),
        zorder     = 5,
    )

    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info("Error plot saved to %s", output_path)


# ============================================================================
# 7. Ponto de entrada
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WxFormer — inferência pontual em dado não visto.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Caminho para best_model.pt (salvo pelo EarlyStopping).",
    )
    p.add_argument(
        "--normalizer", required=True,
        help="Caminho para normalizer.pkl (FeatureNormalizer do treino).",
    )
    p.add_argument(
        "--optuna_db", required=True,
        help="Caminho para optuna.db (gerado em outputs/<timestamp>/).",
    )
    p.add_argument(
        "--gt_nc", required=True,
        help=(
            "Caminho para o .nc do dia alvo contendo apenas t2m. "
            "Usado exclusivamente como ground truth — não como entrada do modelo."
        ),
    )
    p.add_argument(
        "--target_date", default="2026-01-02",
        help="Data de inferência no formato YYYY-MM-DD.",
    )
    p.add_argument(
        "--study_name", default="weather_transformer",
        help="Nome do estudo Optuna (definido em tuning/optuna_search.py).",
    )
    p.add_argument(
        "--output_dir", default="inference_results",
        help="Diretório de saída para PDF e TXT.",
    )
    p.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Dispositivo de inferência.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Estilo paper-like (idêntico ao training/evaluate.py) ─────────────
    _apply_paper_style()

    # ── Dispositivo ───────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    else:
        device = torch.device(args.device)
    log.info("Dispositivo: %s", device)

    # ── Diretório de saída ────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Hiperparâmetros: extrai do banco Optuna ────────────────────────
    hparams = load_hparams_from_db(Path(args.optuna_db), args.study_name)

    # ── 2. FeatureNormalizer do treino ────────────────────────────────────
    normalizer = FeatureNormalizer.load(args.normalizer)
    log.info("FeatureNormalizer carregado de %s", args.normalizer)

    # ── 3. Modelo ─────────────────────────────────────────────────────────
    model = load_model(Path(args.checkpoint), hparams, device)

    # ── 4. Ground truth: t2m real do dia alvo (arquivo NC separado) ───────
    y_true, timestamps = load_ground_truth(Path(args.gt_nc), args.target_date)

    # ── 5. Contexto: features das 168 h anteriores (arquivos de treino) ───
    log.info("Carregando dados de contexto de cfg.NC_FILES ...")
    df_raw = load_and_merge_nc(cfg.NC_FILES)
    df     = build_features(df_raw)

    x_tensor = build_context_window(df, normalizer, args.target_date)

    # ── 6. Inferência ─────────────────────────────────────────────────────
    log.info("Executando inferência para %s ...", args.target_date)
    y_pred = run_inference(model, x_tensor, normalizer, device)

    # ── 7. Métricas ───────────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred)
    log.info(
        "MSE = %.4f °C²  |  RMSE = %.4f °C  |  MAE = %.4f °C",
        metrics["mse"], metrics["rmse"], metrics["mae"],
    )

    # ── 8. Relatório textual ──────────────────────────────────────────────
    txt_path = out_dir / f"metrics_{args.target_date}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("WxFormer — Inferência Pontual\n")
        f.write(f"Data alvo   : {args.target_date}\n")
        f.write(f"Checkpoint  : {args.checkpoint}\n")
        f.write(f"Normalizer  : {args.normalizer}\n")
        f.write(f"Optuna DB   : {args.optuna_db}\n")
        f.write(f"Ground truth: {args.gt_nc}\n")
        f.write("=" * 52 + "\n")
        f.write(f"MSE   = {metrics['mse']:.6f}  °C²\n")
        f.write(f"RMSE  = {metrics['rmse']:.6f}  °C\n")
        f.write(f"MAE   = {metrics['mae']:.6f}  °C\n")
        f.write("=" * 52 + "\n\n")
        f.write(f"{'Timestamp':<22} {'Real (°C)':>10} {'Previsto (°C)':>14} {'Erro (°C)':>10}\n")
        f.write("-" * 58 + "\n")
        for ts, yt, yp in zip(timestamps, y_true, y_pred):
            f.write(f"{str(ts):<22} {yt:>10.4f} {yp:>14.4f} {yp - yt:>10.4f}\n")
    log.info("Relatório textual salvo em %s", txt_path)

    # ── 9. Figuras (dois PDFs separados) ─────────────────────────────────
    temp_path  = out_dir / f"inference_temperature_{args.target_date}.pdf"
    error_path = out_dir / f"inference_error_{args.target_date}.pdf"

    plot_temperature(y_true, y_pred, args.target_date, temp_path)
    plot_hourly_error(y_true, y_pred, metrics, args.target_date, error_path)

    log.info("Concluido. Resultados em: %s/", out_dir)


if __name__ == "__main__":
    main()