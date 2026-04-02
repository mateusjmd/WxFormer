"""
explainability/explain.py
=========================
Etapa 13 do pipeline: explicabilidade do modelo.

Dois métodos são implementados:

1. Mapa de Atenção (última camada do encoder)
   Extrai os pesos de atenção via forward hook e os visualiza como
   mapa de calor entre patches temporais.

2. Saliência por Gradiente × Input
   Calcula |d(pred)/d(X)| × |X| para cada feature em cada passo de
   tempo, quantificando a sensibilidade da previsão a perturbações
   locais.

Estilo dos Plots (paper-like)
------------------------------
Os gráficos seguem as mesmas convenções paper-like definidas em
training/evaluate.py (rcParams configurados globalmente ao importar
aquele módulo). Caso explain.py seja usado isoladamente, os rcParams
do Matplotlib padrão são usados — recomenda-se importar evaluate.py
antes, ou chamar _apply_paper_style() diretamente.

Referências:
  - Simonyan, K. et al. (2014). Deep Inside Convolutional Networks.
    arXiv:1312.6034
  - Shrikumar, A. et al. (2017). DeepLIFT. ICML 2017.
  - Abnar, S. & Zuidema, W. (2020). Quantifying Attention Flow.
    ACL 2020. arXiv:2005.00928
  - Jain, S. & Wallace, B. C. (2019). Attention is not explanation.
    NAACL 2019.
"""

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook para captura de pesos de atenção
# ---------------------------------------------------------------------------

class AttentionHook:
    """
    Hook de forward que intercepta a saída de nn.MultiheadAttention
    e armazena os pesos de atenção média sobre cabeças.

    Uso:
        hook   = AttentionHook()
        handle = layer.self_attn.register_forward_hook(hook)
        model(x)
        attn   = hook.weights   # Tensor (B, P, P)
        handle.remove()
    """

    def __init__(self):
        self.weights = None

    def __call__(self, module, inputs, outputs):
        # outputs de MHA: (attn_output, attn_weights)
        # attn_weights só está presente quando need_weights=True
        if isinstance(outputs, tuple) and len(outputs) == 2:
            self.weights = outputs[1]  # (B, P, P), média sobre heads


# ---------------------------------------------------------------------------
# Extração de pesos de atenção
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_attention_weights(
    model:      nn.Module,
    x:          torch.Tensor,
    layer_idx:  int = -1,
) -> np.ndarray:
    """
    Extrai pesos de atenção da camada `layer_idx` do encoder.

    Parâmetros
    ----------
    model     : WeatherTransformer
    x         : Tensor, shape (1, L, F) ou (L, F)
    layer_idx : int — índice da camada (default: -1 = última)

    Retorna
    -------
    np.ndarray, shape (P, P) — pesos médios (sobre heads)
    """
    model.eval()
    if x.dim() == 2:
        x = x.unsqueeze(0)  # adiciona batch dim

    target_layer  = model.encoder_layers[layer_idx]
    hook          = AttentionHook()
    handle        = target_layer.self_attn.register_forward_hook(hook)
    original_fw   = target_layer.self_attn.forward

    # Monkey-patch temporário para forçar need_weights=True
    def _fw_with_weights(query, key, value, **kwargs):
        kwargs["need_weights"]         = True
        kwargs["average_attn_weights"] = True
        return original_fw(query, key, value, **kwargs)

    target_layer.self_attn.forward = _fw_with_weights

    try:
        _ = model(x)
        attn = hook.weights
    finally:
        target_layer.self_attn.forward = original_fw
        handle.remove()

    if attn is None:
        raise RuntimeError("Nao foi possivel capturar os pesos de atencao.")

    return attn.squeeze(0).cpu().numpy()  # (P, P)


# ---------------------------------------------------------------------------
# Saliência por Gradiente × Input
# ---------------------------------------------------------------------------

def compute_gradient_saliency(
    model:       nn.Module,
    x:           torch.Tensor,
    target_step: int          = 0,
    device:      torch.device = None,
) -> np.ndarray:
    """
    Calcula o mapa de saliência Gradient × Input.

    saliencia(t, f) = |d(pred[target_step]) / d(X[t,f])| × |X[t,f]|

    Parâmetros
    ----------
    model       : WeatherTransformer
    x           : Tensor, shape (L, F)
    target_step : int — passo do horizonte a analisar (0 = 1ª hora)
    device      : torch.device

    Retorna
    -------
    np.ndarray, shape (L, F)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    x_input = x.unsqueeze(0).to(device).float()
    x_input.requires_grad_(True)

    pred = model(x_input)  # (1, horizon)
    model.zero_grad()
    pred[0, target_step].backward()

    grad     = x_input.grad.detach().cpu()
    x_np     = x_input.detach().cpu()
    saliency = (grad.abs() * x_np.abs()).squeeze(0).numpy()

    return saliency  # (L, F)


# ---------------------------------------------------------------------------
# Visualizações paper-like
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    attn_weights: np.ndarray,
    patch_size:   int        = cfg.PATCH_SIZE,
    save_path:    str | None = None,
) -> None:
    """
    Mapa de calor dos pesos de atenção entre patches temporais.

    Layout de 1 coluna IEEE (3.5 in). Eixos rotulados em horas
    relativas ao inicio da janela de contexto.

    Parâmetros
    ----------
    attn_weights : np.ndarray, shape (P, P)
    patch_size   : int — duração de cada patch em horas
    save_path    : str | None
    """
    P = attn_weights.shape[0]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    im = ax.imshow(
        attn_weights,
        cmap   = "Blues",        # sequencial, sem saturação em extremos
        aspect = "auto",
        vmin   = 0.0,
        vmax   = attn_weights.max(),
        interpolation = "nearest",
    )

    # Colorbar com tamanho proporcional ao eixo
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention weight", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Rótulos: mostrar apenas a cada 4 patches para evitar sobreposição
    tick_positions = np.arange(0, P, 4)
    tick_labels    = [f"{int(i * patch_size)}h" for i in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)

    ax.set_xlabel("Key patch (time step)")
    ax.set_ylabel("Query patch (time step)")

    # Linha diagonal tracejada para referência (auto-atenção)
    ax.plot([0, P - 1], [0, P - 1], color="white", linewidth=0.6,
            linestyle="--", alpha=0.6)

    if save_path:
        fig.savefig(save_path)
        logger.info("Attention heatmap saved to %s", save_path)

    plt.close(fig)


def plot_saliency_heatmap(
    saliency:      np.ndarray,
    feature_names: list[str],
    save_path:     str | None = None,
) -> None:
    """
    Mapa de saliência Gradient × Input (features × tempo).

    Eixo horizontal: passo de tempo na janela de contexto (horas).
    Eixo vertical: features de entrada, ordenadas por saliência média
    descendente para facilitar a leitura.

    Layout de 2 colunas IEEE (7.0 in).

    Parâmetros
    ----------
    saliency      : np.ndarray, shape (L, F)
    feature_names : list[str]
    save_path     : str | None
    """
    L, F = saliency.shape

    # Ordena features por saliência média (descendente) para hierarquia visual
    mean_sal    = saliency.mean(axis=0)       # (F,)
    order       = np.argsort(mean_sal)[::-1]  # índices do maior para menor
    sal_sorted  = saliency[:, order].T        # (F, L) — linhas = features
    names_sorted = [feature_names[i] for i in order]

    # Normalização para [0, 1] — importância relativa entre features
    sal_norm = sal_sorted / (sal_sorted.max() + 1e-10)

    fig, ax = plt.subplots(figsize=(7.0, 0.35 * F + 0.6))

    im = ax.imshow(
        sal_norm,
        cmap          = "YlOrRd",   # sequencial, intuitivo para "intensidade"
        aspect        = "auto",
        vmin          = 0.0,
        vmax          = 1.0,
        interpolation = "nearest",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Normalized saliency", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Eixo x: horas relativas ao momento da previsão (negativo = passado)
    step = max(1, L // 8)          # ~8 ticks no eixo
    x_ticks = np.arange(0, L, step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [f"$-${L - t}h" for t in x_ticks],
        rotation = 45,
        ha       = "right",
        fontsize = 8,
    )

    ax.set_yticks(range(F))
    ax.set_yticklabels(names_sorted, fontsize=9)

    ax.set_xlabel("Context window (hours before forecast)")
    ax.set_ylabel("Input feature")

    # Linha vertical tracejada: limite entre passado recente e remoto
    mid = L // 2
    ax.axvline(mid, color="white", linewidth=0.8, linestyle=":", alpha=0.7)

    if save_path:
        fig.savefig(save_path)
        logger.info("Saliency heatmap saved to %s", save_path)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline completo de explicabilidade
# ---------------------------------------------------------------------------

def explain_sample(
    model:          nn.Module,
    x_sample:       torch.Tensor,
    feature_names:  list[str],
    device:         torch.device,
    output_dir:     str = "outputs",
) -> None:
    """
    Gera e salva todas as visualizações de explicabilidade para
    uma amostra de entrada.

    Parâmetros
    ----------
    model         : WeatherTransformer
    x_sample      : Tensor, shape (L, F)
    feature_names : list[str]
    device        : torch.device
    output_dir    : str — diretório timestampado de saída
    """
    logger.info("Generating explainability visualizations...")

    # 1. Pesos de atenção (última camada)
    attn = extract_attention_weights(model, x_sample.to(device))
    plot_attention_heatmap(
        attn,
        save_path = f"{output_dir}/attention_heatmap.pdf",
    )

    # 2. Saliência Gradient × Input (1ª hora do horizonte)
    saliency = compute_gradient_saliency(
        model, x_sample, target_step=0, device=device
    )
    plot_saliency_heatmap(
        saliency,
        feature_names = feature_names,
        save_path     = f"{output_dir}/saliency_heatmap.pdf",
    )

    logger.info("Explainability figures saved to '%s/'.", output_dir)
