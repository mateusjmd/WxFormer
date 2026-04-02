"""
model/scheduler.py
==================
Etapa 11 do pipeline: schedulers de learning rate.

Duas estratégias são implementadas:
  1. CosineAnnealingLR — decaimento cossenoidal puro
  2. CosineAnnealingLR + Linear Warmup — aquecimento linear seguido
     de decaimento cossenoidal

Motivação
---------
O decaimento cossenoidal foi proposto por Loshchilov & Hutter (2017) e
demonstrou superar o decaimento linear e em escada em diversas tarefas
de deep learning.  O warmup linear (Goyal et al., 2017) estabiliza o
treinamento nas épocas iniciais quando os parâmetros ainda estão longe
do ótimo e os gradientes são ruidosos.

Referências:
  - Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic Gradient
    Descent with Warm Restarts. ICLR 2017. arXiv:1608.03983
  - Goyal, P. et al. (2017). Accurate, Large Minibatch SGD: Training
    ImageNet in 1 Hour. arXiv:1706.02677
  - Liu, X. et al. (2020). On the Variance of the Adaptive Learning
    Rate and Beyond. ICLR 2020. (RAdam / warmup theory)
"""

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


# ---------------------------------------------------------------------------
# Cosine com Warmup (scheduler baseado em lambda)
# ---------------------------------------------------------------------------

def get_cosine_warmup_scheduler(
    optimizer:     Optimizer,
    warmup_steps:  int,
    total_steps:   int,
    eta_min_ratio: float = 0.01,
) -> LambdaLR:
    """
    Scheduler: aquecimento linear por warmup_steps passos,
    seguido de decaimento cossenoidal até eta_min_ratio * lr_base.

        lr(t) =
            t / warmup_steps * lr_base,           se t < warmup_steps
            eta_min + 0.5*(lr_base - eta_min) *
            (1 + cos(π*(t-warmup)/(total-warmup))), caso contrário

    Parâmetros
    ----------
    optimizer     : Optimizer — otimizador PyTorch
    warmup_steps  : int  — número de passos do warmup linear
    total_steps   : int  — número total de passos de treinamento
    eta_min_ratio : float — fração do lr_base como lr mínimo

    Retorna
    -------
    LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Fase de warmup linear
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Fase cossenoidal
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Interpolação entre eta_min e 1.0 (multiplicador sobre lr_base)
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_scheduler(
    optimizer:   Optimizer,
    total_steps: int,
    eta_min_ratio: float = 0.01,
) -> LambdaLR:
    """
    Scheduler cossenoidal puro (sem warmup).

    Parâmetros
    ----------
    optimizer     : Optimizer
    total_steps   : int   — total de passos de treinamento
    eta_min_ratio : float — fração do lr_base como lr mínimo

    Retorna
    -------
    LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        progress     = float(current_step) / float(max(1, total_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Fábrica de schedulers
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer:     Optimizer,
    scheduler_type: str,
    total_steps:   int,
    warmup_steps:  int  = 0,
    eta_min_ratio: float = 0.01,
) -> LambdaLR:
    """
    Fábrica que constrói o scheduler correto com base no nome.

    Parâmetros
    ----------
    optimizer      : Optimizer
    scheduler_type : str  — 'cosine' ou 'cosine_warmup'
    total_steps    : int  — total de steps de treinamento
    warmup_steps   : int  — steps de warmup (ignorado para 'cosine')
    eta_min_ratio  : float

    Retorna
    -------
    LambdaLR scheduler

    Raises
    ------
    ValueError se scheduler_type inválido
    """
    if scheduler_type == "cosine":
        return get_cosine_scheduler(optimizer, total_steps, eta_min_ratio)

    elif scheduler_type == "cosine_warmup":
        return get_cosine_warmup_scheduler(
            optimizer, warmup_steps, total_steps, eta_min_ratio
        )

    else:
        raise ValueError(
            f"scheduler_type inválido: '{scheduler_type}'. "
            "Opções: 'cosine', 'cosine_warmup'."
        )
