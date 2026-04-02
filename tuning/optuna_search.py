"""
tuning/optuna_search.py
=======================
Otimização de hiperparâmetros com Optuna usando o TPE Sampler
(Tree-structured Parzen Estimator).

Por que Optuna / TPE?
---------------------
A busca aleatória e em grade são estratégias agnósticas: não aprendem
com trials anteriores.  O TPE é um algoritmo Bayesiano que modela a
distribuição de hiperparâmetros condicional à qualidade do objetivo
(Bergstra et al., 2011).  Em cada trial, ele ajusta dois modelos de
densidade — um para bons resultados e outro para ruins — e propõe
configurações que maximizam a razão p_good / p_bad.

Isso resulta em convergência muito mais rápida que busca aleatória,
especialmente quando o espaço de busca é grande ou há interações
entre hiperparâmetros.

Pruning (CMA-ES Media Pruner)
-------------------------------
Optuna suporta pruning de trials sem potencial: o MedianPruner
interrompe trials cujo resultado intermediário é pior que a mediana
dos trials já completos até aquele ponto.  Isso economiza até 60%
do tempo de busca sem afetar a qualidade da solução final
(Li et al., 2018).

Espaço de Busca e Justificativas
----------------------------------
  n_layers ∈ [2, 8]:
    Transformers rasos (2-3 camadas) generalizam bem para séries
    temporais curtas. Camadas adicionais capturam padrões hierárquicos
    mas aumentam o risco de overfitting (Nie et al., 2023).

  d_model ∈ {128, 256, 512}:
    Múltiplos de potências de 2 facilitam o alinhamento de memória nas
    GPUs (cudnn convolution kernel efficiency). 512 é equivalente ao
    BERT-base (Devlin et al., 2019), escolhido como teto razoável para
    o tamanho do dataset (~26k amostras de treino).

  n_heads ∈ {4, 8} (com restrição d_model % n_heads == 0):
    Cabeças múltiplas permitem ao modelo atender a diferentes
    sub-espaços de atenção simultaneamente (Vaswani et al., 2017).
    4 e 8 são valores canônicos; mais de 8 raramente traz ganho em
    séries temporais com d_model ≤ 512.

  ff_mult ∈ [2, 6] → dim_ff = ff_mult * d_model:
    O Transformer original usa dim_ff = 4 * d_model.  A faixa [2, 6]
    explora configurações mais estreitas (menos parâmetros, mais
    regularização implícita) e mais largas (maior capacidade).

  attn_dropout ∈ [0.0, 0.3]:
    Dropout de atenção acima de 0.3 tende a degradar a convergência
    em Transformers de encoder (Zehui et al., 2019).

  ff_dropout ∈ [0.0, 0.4]:
    Dropout na MLP feedforward pode ser levemente maior, pois a MLP
    tem mais parâmetros e maior tendência ao overfitting.

  learning_rate ∈ [1e-5, 3e-3] (escala log):
    A escala logarítmica é necessária porque a importância relativa
    de diferenças na LR varia em ordens de magnitude.  O AdamW com
    warmup tolera LRs maiores que o SGD (Devlin et al., 2019).

  weight_decay ∈ [1e-6, 1e-2] (escala log):
    Regularização L2 via weight_decay do AdamW, que desacopla o
    decaimento de peso da atualização do gradiente adaptativo
    (Loshchilov & Hutter, 2019).

  batch_size ∈ {64, 128, 256}:
    Batch maior → gradiente menos ruidoso, mas menor regularização
    implícita. As três opções cobrem o trade-off relevante para o
    dataset (~26k amostras de treino).

  scheduler ∈ {cosine, cosine_warmup}:
    Ambos são estratégias state-of-the-art; o warmup é especialmente
    benéfico quando a LR inicial é alta (Goyal et al., 2017).

  warmup_steps ∈ [0, 2000]:
    Com batch_size=128 e ~26k amostras, 1 época ≈ 204 steps.
    2000 steps ≈ 10 épocas de warmup, que é o teto razoável
    recomendado por Liu et al. (2020).

Referências:
  - Bergstra, J. & Bengio, Y. (2012). Random search for hyper-parameter
    optimization. JMLR, 13, 281-305.
  - Bergstra, J. et al. (2011). Algorithms for hyper-parameter
    optimization. NeurIPS 2011.
  - Li, L. et al. (2018). Hyperband: A novel bandit-based approach to
    hyperparameter optimization. JMLR, 18(1), 6765-6816.
  - Akiba, T. et al. (2019). Optuna: A next-generation hyperparameter
    optimization framework. KDD 2019.
  - Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay
    Regularization. ICLR 2019. arXiv:1711.05101
"""

import logging
import os

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cfg
from model.transformer import WeatherTransformer
from model.scheduler import build_scheduler
from training.trainer import train_epoch, validate_epoch, EarlyStopping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Função objetivo do Optuna
# ---------------------------------------------------------------------------

def objective(
    trial:        optuna.Trial,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
) -> float:
    """
    Função objetivo: treina um modelo com a configuração proposta pelo
    trial e retorna a melhor loss de validação.

    O retorno é a loss MSE normalizada (não em °C), o que é correto
    para a otimização interna do Optuna — a escala absoluta não importa
    porque todos os trials usam o mesmo normalizer.

    Parâmetros
    ----------
    trial        : optuna.Trial  — objeto de trial com os hiperparâmetros
    train_loader : DataLoader
    val_loader   : DataLoader
    device       : torch.device

    Retorna
    -------
    float — melhor val_loss do trial (MSE normalizado)
    """
    # ---- Amostragem de hiperparâmetros --------------------------------

    # Profundidade do encoder: mais camadas = mais capacidade
    n_layers = trial.suggest_int("n_layers", 2, 8)

    # Dimensão do modelo: potências de 2 para eficiência em GPU
    d_model  = trial.suggest_categorical("d_model", [128, 256, 512])

    # Número de cabeças: restrito para que d_model seja divisível por n_heads
    valid_heads = [h for h in [4, 8] if d_model % h == 0]
    n_heads     = trial.suggest_categorical("n_heads", valid_heads)

    # Multiplicador da dimensão feedforward
    ff_mult = trial.suggest_int("ff_mult", 2, 6)
    dim_ff  = ff_mult * d_model

    # Dropouts separados por sub-camada
    attn_dropout = trial.suggest_float("attn_dropout", 0.0, 0.3)
    ff_dropout   = trial.suggest_float("ff_dropout",   0.0, 0.4)

    # Hiperparâmetros do otimizador
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    weight_decay  = trial.suggest_float("weight_decay",  1e-6, 1e-2, log=True)

    # Batch size (atualiza o DataLoader dinamicamente é inviável;
    # aqui usamos o loader pré-criado; batch_size é amostrado mas não
    # aplicado ao loader existente — veja main.py para uso completo)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # Scheduler
    scheduler_type = trial.suggest_categorical(
        "scheduler", ["cosine", "cosine_warmup"]
    )
    warmup_steps = (
        trial.suggest_int("warmup_steps", 0, 2000)
        if scheduler_type == "cosine_warmup"
        else 0
    )

    # ---- Construção do modelo ------------------------------------------
    model = WeatherTransformer(
        input_dim    = cfg.INPUT_DIM,
        d_model      = d_model,
        n_heads      = n_heads,
        n_layers     = n_layers,
        dim_ff       = dim_ff,
        attn_dropout = attn_dropout,
        ff_dropout   = ff_dropout,
        horizon      = cfg.HORIZON,
        seq_len      = cfg.WINDOW,
        patch_size   = cfg.PATCH_SIZE,
        use_alibi    = True,
    ).to(device)

    logger.debug(
        "Trial %d: n_layers=%d, d_model=%d, n_heads=%d, dim_ff=%d, "
        "attn_drop=%.3f, ff_drop=%.3f, lr=%.2e, wd=%.2e, sched=%s, "
        "params=%d",
        trial.number, n_layers, d_model, n_heads, dim_ff,
        attn_dropout, ff_dropout, learning_rate, weight_decay,
        scheduler_type, model.count_parameters(),
    )

    # ---- Otimizador e scheduler ----------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = learning_rate,
        weight_decay = weight_decay,
    )

    # Total de steps = épocas * batches por época
    total_steps = cfg.OPTUNA_EPOCHS * len(train_loader)

    scheduler = build_scheduler(
        optimizer      = optimizer,
        scheduler_type = scheduler_type,
        total_steps    = total_steps,
        warmup_steps   = warmup_steps,
    )

    # ---- Treinamento com Early Stopping --------------------------------
    criterion   = nn.MSELoss()
    early_stop  = EarlyStopping(
        patience        = cfg.OPTUNA_PATIENCE,
        checkpoint_path = f"checkpoints/trial_{trial.number}.pt",
    )

    best_val = float("inf")

    for epoch in range(1, cfg.OPTUNA_EPOCHS + 1):
        train_epoch(
            model, train_loader, optimizer, criterion,
            device, scheduler, max_norm=cfg.GRAD_CLIP_MAX_NORM,
        )
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Reporta progresso intermediário ao Optuna (habilita pruning)
        trial.report(val_loss, epoch)

        # Pruna trial sem potencial (MedianPruner)
        if trial.should_prune():
            logger.debug("Trial %d podado na época %d.", trial.number, epoch)
            raise optuna.exceptions.TrialPruned()

        # Early stopping interno do trial
        if early_stop(val_loss, model):
            break

        best_val = min(best_val, early_stop.best_loss)

    # Remove checkpoint temporário do trial
    ckpt_path = f"checkpoints/trial_{trial.number}.pt"
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return early_stop.best_loss


# ---------------------------------------------------------------------------
# Execução do estudo Optuna
# ---------------------------------------------------------------------------

def run_optuna_study(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    n_trials:     int = cfg.N_TRIALS,
    study_name:   str = "weather_transformer",
    storage:      str | None = None,
) -> optuna.Study:
    """
    Executa o estudo Optuna com TPE Sampler e MedianPruner.

    Parâmetros
    ----------
    train_loader : DataLoader
    val_loader   : DataLoader
    device       : torch.device
    n_trials     : int  — número de trials (default: 100)
    study_name   : str  — nome do estudo (para logging e storage)
    storage      : str  — URI do banco de dados para persistência
                          (ex.: "sqlite:///optuna.db"). None = in-memory.

    Retorna
    -------
    optuna.Study  — estudo concluído com todos os trials e melhor trial
    """
    # TPE Sampler: algoritmo Bayesiano baseado em estimativas de densidade
    sampler = TPESampler(
        seed              = cfg.SEED,
        n_startup_trials  = 10,   # trials iniciais aleatórios para warm-up
        multivariate      = True,  # modelagem de correlações entre HPs
        consider_prior    = True,  # usa prior uniforme como regularizador
    )

    # MedianPruner: interrompe trials com desempenho abaixo da mediana
    pruner = MedianPruner(
        n_startup_trials  = 10,  # não pruna nos primeiros 10 trials
        n_warmup_steps    = 5,   # aguarda 5 épocas antes de podar
        interval_steps    = 1,   # verifica a cada época
    )

    study = optuna.create_study(
        study_name     = study_name,
        direction      = "minimize",      # minimiza val MSE
        sampler        = sampler,
        pruner         = pruner,
        storage        = storage,
        load_if_exists = True,            # retoma estudo se já existir
    )

    logger.info(
        "Iniciando estudo Optuna com %d trials | TPE + MedianPruner",
        n_trials,
    )

    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device),
        n_trials  = n_trials,
        timeout   = None,
        show_progress_bar = True,
        gc_after_trial    = True,   # libera memória GPU após cada trial
    )

    # Relatório final
    best = study.best_trial
    logger.info("=" * 60)
    logger.info("MELHOR TRIAL: #%d", best.number)
    logger.info("  Val MSE: %.6f", best.value)
    for k, v in best.params.items():
        logger.info("  %s: %s", k, v)
    logger.info("=" * 60)

    return study


def best_hyperparams_from_study(study: optuna.Study) -> dict:
    """
    Extrai os hiperparâmetros do melhor trial para uso no treinamento final.

    Parâmetros
    ----------
    study : optuna.Study  — estudo concluído

    Retorna
    -------
    dict com todos os hiperparâmetros do melhor trial
    """
    best = study.best_trial
    params = best.params.copy()

    # Recalcula dim_ff a partir de ff_mult e d_model
    params["dim_ff"] = params.pop("ff_mult") * params["d_model"]

    return params
