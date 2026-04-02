"""
training/evaluate.py
====================
Etapa 12 do pipeline: avaliação no conjunto de teste e métricas.

Métricas implementadas:
  - MSE  : Mean Squared Error (na escala normalizada e em °C²)
  - RMSE : Root Mean Squared Error (em °C)
  - MAE  : Mean Absolute Error (em °C)

Estilo dos Plots (paper-like)
------------------------------
Todos os gráficos seguem convenções de publicação científica:
  - Fonte: serif (Times-compatible) via rcParams
  - DPI 300 para impressão de alta qualidade
  - Linhas finas com marcadores discretos
  - Espessura de eixos e ticks aumentada (spines, tick_params)
  - Sem título de figura (caption inserida externamente no LaTeX/Word)
  - Legendas sem borda excessiva (framealpha=0.8)
  - Paleta de cores diferenciável em escala de cinza (acessibilidade)
  - Formato .pdf (vetorial) quando possível; .png com dpi=300 alternativa

Referências:
  - Rasp, S. et al. (2020). WeatherBench. JAMES, 12(11).
  - Chai, T. & Draxler, R. R. (2014). RMSE or MAE?
    Geoscientific Model Development, 7(3), 1247–1250.
  - Tufte, E. R. (2001). The Visual Display of Quantitative Information.
    Graphics Press. (princípios de visualização científica)
"""

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tema global paper-like (aplicado uma vez na importação do módulo)
# ---------------------------------------------------------------------------

def _apply_paper_style() -> None:
    """
    Configura rcParams do Matplotlib para estilo compatível com
    publicações científicas (IEEE / AGU / AMS).

    As configurações são aplicadas globalmente no processo Python,
    afetando todos os gráficos gerados após esta chamada.
    """
    mpl.rcParams.update({
        # Fonte principal: DejaVu Serif aproxima Times New Roman em LaTeX
        "font.family":          "serif",
        "font.serif":           ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size":            10,          # corpo do texto

        # Tamanho específico por elemento
        "axes.titlesize":       11,
        "axes.labelsize":       10,
        "xtick.labelsize":       9,
        "ytick.labelsize":       9,
        "legend.fontsize":       9,

        # Resolução de saída
        "figure.dpi":          150,          # preview em tela
        "savefig.dpi":         300,          # impressão/publicação

        # Espessura de elementos estruturais
        "axes.linewidth":        0.8,        # borda dos eixos
        "xtick.major.width":     0.8,
        "ytick.major.width":     0.8,
        "xtick.minor.width":     0.5,
        "ytick.minor.width":     0.5,
        "xtick.major.size":      4.0,
        "ytick.major.size":      4.0,
        "xtick.direction":      "in",        # ticks para dentro (padrão AGU)
        "ytick.direction":      "in",
        "xtick.top":             True,        # ticks nos 4 lados
        "ytick.right":           True,

        # Linhas dos plots
        "lines.linewidth":       1.4,
        "lines.markersize":      4.5,

        # Grid discreto
        "axes.grid":             True,
        "grid.linewidth":        0.4,
        "grid.alpha":            0.4,
        "grid.linestyle":       "--",

        # Paleta de cores acessível (distinguível em P&B e por daltônicos)
        # Baseada na paleta Paul Tol "bright"
        "axes.prop_cycle": mpl.cycler(color=[
            "#4477AA",   # azul
            "#EE6677",   # vermelho-rosa
            "#228833",   # verde
            "#CCBB44",   # amarelo-ocre
            "#66CCEE",   # ciano
            "#AA3377",   # roxo
            "#BBBBBB",   # cinza
        ]),

        # Legenda limpa
        "legend.framealpha":     0.85,
        "legend.edgecolor":     "0.8",
        "legend.borderpad":      0.4,

        # Layout e margens
        "figure.constrained_layout.use": True,

        # Formato vetorial padrão para salvar
        "savefig.format":       "pdf",
        "savefig.bbox":         "tight",
        "savefig.pad_inches":    0.05,
    })


# Aplica o tema no momento da importação do módulo
_apply_paper_style()


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calcula MSE, RMSE e MAE entre valores reais e previstos.

    R² foi deliberadamente omitido — ver docstring do módulo.

    Parâmetros
    ----------
    y_true : np.ndarray — valores reais em °C (escala original)
    y_pred : np.ndarray — previsões em °C

    Retorna
    -------
    dict com chaves: 'mse', 'rmse', 'mae'
    """
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(y_true - y_pred)))

    return {"mse": mse, "rmse": rmse, "mae": mae}


# ---------------------------------------------------------------------------
# Inferência no conjunto de teste
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model:      nn.Module,
    loader:     DataLoader,
    normalizer,               # FeatureNormalizer
    device:     torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Realiza inferência no conjunto de teste e inverte a normalização.

    Parâmetros
    ----------
    model      : nn.Module
    loader     : DataLoader de teste
    normalizer : FeatureNormalizer — contém o target_scaler
    device     : torch.device

    Retorna
    -------
    (y_true_orig, y_pred_orig) em °C, shape (N, horizon)
    """
    model.eval()
    all_preds = []
    all_trues = []

    for X, y in loader:
        X = X.to(device)
        pred = model(X)
        all_preds.append(pred.cpu().numpy())
        all_trues.append(y.numpy())

    y_pred_norm = np.concatenate(all_preds, axis=0)
    y_true_norm = np.concatenate(all_trues, axis=0)

    y_pred_orig = normalizer.inverse_transform_target(y_pred_norm)
    y_true_orig = normalizer.inverse_transform_target(y_true_norm)

    return y_true_orig, y_pred_orig


def evaluate_on_test(
    model:      nn.Module,
    loader:     DataLoader,
    normalizer,
    device:     torch.device,
) -> dict[str, float]:
    """
    Avalia o modelo no conjunto de teste, loga e retorna as métricas.

    Retorna
    -------
    dict com 'mse', 'rmse', 'mae' na escala original (°C / °C²)
    """
    y_true, y_pred = predict(model, loader, normalizer, device)
    metrics = compute_metrics(y_true.ravel(), y_pred.ravel())

    logger.info("=" * 50)
    logger.info("AVALIACAO NO CONJUNTO DE TESTE")
    logger.info("  MSE  = %.4f graus^2", metrics["mse"])
    logger.info("  RMSE = %.4f graus C", metrics["rmse"])
    logger.info("  MAE  = %.4f graus C", metrics["mae"])
    logger.info("=" * 50)

    return metrics


# ---------------------------------------------------------------------------
# Visualizações paper-like
# ---------------------------------------------------------------------------

def plot_learning_curves(
    train_losses: list[float],
    val_losses:   list[float],
    save_path:    str | None = None,
) -> None:
    """
    Curvas de aprendizado: MSE normalizado vs. época.

    Layout de coluna simples (largura ~8,8 cm = 1 coluna IEEE).
    Sem título de figura — caption deve ser adicionada externamente.

    Parâmetros
    ----------
    train_losses : list[float] — MSE de treino por época
    val_losses   : list[float] — MSE de validação por época
    save_path    : str | None  — caminho de saída (.pdf ou .png)
    """
    # Largura de 1 coluna IEEE ≈ 3.5 in; altura compatível com relação 4:3
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    epochs  = np.arange(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses,
            label     = "Training",
            linestyle = "-",
            marker    = "",           # sem marcadores em séries longas
            linewidth = 1.4)

    ax.plot(epochs, val_losses,
            label     = "Validation",
            linestyle = "--",
            marker    = "",
            linewidth = 1.4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalized scale)")
    ax.legend(loc="upper right")

    # Remove spine superior e direito (estilo Tufte minimalista)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path:
        fig.savefig(save_path)
        logger.info("Learning curves saved to %s", save_path)

    plt.close(fig)


def plot_predictions(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    n_samples: int       = 3,
    save_path: str | None = None,
) -> None:
    """
    Painéis de comparação previsão vs. observação para n_samples
    sequências do conjunto de teste.

    Cada painel cobre o horizonte completo de 24 h. As amostras são
    selecionadas em intervalos regulares ao longo do conjunto de teste
    para representar sazonalidade intra-anual.

    Layout de 2 colunas IEEE (largura ~17,8 cm).

    Parâmetros
    ----------
    y_true    : np.ndarray, shape (N, horizon) — observações em °C
    y_pred    : np.ndarray, shape (N, horizon) — previsões em °C
    n_samples : int  — número de sequências
    save_path : str | None
    """
    # Largura de 2 colunas IEEE ≈ 7.0 in
    fig, axes = plt.subplots(
        n_samples, 1,
        figsize    = (7.0, 2.0 * n_samples),
        sharex     = True,
    )

    # Garante que axes seja sempre iterável
    if n_samples == 1:
        axes = [axes]

    indices = np.linspace(0, len(y_true) - 1, n_samples, dtype=int)
    hours   = np.arange(1, y_true.shape[1] + 1)

    for k, (ax, idx) in enumerate(zip(axes, indices)):
        # Sombreamento do intervalo de erro
        ax.fill_between(
            hours,
            y_true[idx],
            y_pred[idx],
            alpha = 0.15,
            color = "#4477AA",
            label = "_nolegend_",
        )

        ax.plot(hours, y_true[idx],
                label     = "Observed",
                linestyle = "-",
                marker    = "o",
                markersize = 2.5,
                markerfacecolor = "white",
                markeredgewidth = 0.8,
                linewidth = 1.2)

        ax.plot(hours, y_pred[idx],
                label     = "Forecast",
                linestyle = "--",
                marker    = "s",
                markersize = 2.5,
                markerfacecolor = "white",
                markeredgewidth = 0.8,
                linewidth = 1.2)

        # Calcula RMSE e MAE desta sequência para anotação no painel
        seq_rmse = float(np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2)))
        seq_mae  = float(np.mean(np.abs(y_true[idx] - y_pred[idx])))

        ax.annotate(
            f"RMSE = {seq_rmse:.2f} °C  |  MAE = {seq_mae:.2f} °C",
            xy         = (0.02, 0.93),
            xycoords   = "axes fraction",
            fontsize   = 8,
            va         = "top",
            ha         = "left",
            bbox       = dict(boxstyle="round,pad=0.2", fc="white",
                              ec="0.8", alpha=0.85),
        )

        ax.set_ylabel("Temperature (°C)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if k == 0:
            ax.legend(loc="upper right", ncol=2)

    axes[-1].set_xlabel("Forecast horizon (h)")

    # Ticks no eixo x apenas no último painel (sharex=True)
    axes[-1].set_xticks(np.arange(0, y_true.shape[1] + 1, 6))

    if save_path:
        fig.savefig(save_path)
        logger.info("Prediction panels saved to %s", save_path)

    plt.close(fig)
