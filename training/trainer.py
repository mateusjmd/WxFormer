"""
training/trainer.py
===================
Etapas 10 e 11 do pipeline: loop de treinamento, Early Stopping e
scheduler de learning rate.

Early Stopping
--------------
Monitora a loss de validação e interrompe o treinamento quando não há
melhora por `patience` épocas consecutivas. Salva o checkpoint do
melhor modelo para posterior recuperação.

Gradient Clipping
-----------------
`torch.nn.utils.clip_grad_norm_` limita a norma L2 do gradiente
acumulado a `max_norm=1.0`. Isso previne explosão de gradientes, que é
frequente em Transformers profundos (Pascanu et al., 2013).

Referências:
  - Prechelt, L. (1998). Early stopping — But when? In: Neural Networks:
    Tricks of the Trade. Springer. pp. 55–69.
  - Pascanu, R. et al. (2013). On the difficulty of training recurrent
    neural networks. ICML 2013. arXiv:1211.5063
"""

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Critério de parada antecipada baseado na loss de validação.

    Se a loss de validação não melhorar em `patience` épocas
    consecutivas, o treinamento é interrompido e o melhor checkpoint
    é restaurado.

    Parâmetros
    ----------
    patience  : int   — épocas sem melhora antes de parar
    min_delta : float — melhora mínima considerada como progresso
    checkpoint_path : str — onde salvar o melhor modelo
    """

    def __init__(
        self,
        patience:         int   = 10,
        min_delta:        float = 1e-4,
        checkpoint_path:  str   = "checkpoints/best_model.pt",
    ):
        self.patience        = patience
        self.min_delta       = min_delta
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_loss   = math.inf
        self.wait_count  = 0
        self.should_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Verifica se o treinamento deve ser interrompido.

        Parâmetros
        ----------
        val_loss : float      — loss de validação da época atual
        model    : nn.Module  — modelo a ser salvo se houver melhora

        Retorna
        -------
        bool — True se deve parar o treinamento
        """
        if val_loss < self.best_loss - self.min_delta:
            # Houve melhora: salva checkpoint e reinicia contador
            self.best_loss  = val_loss
            self.wait_count = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            logger.debug(
                "Melhor val_loss atualizado: %.6f → checkpoint salvo", val_loss
            )
        else:
            # Sem melhora: incrementa contador de espera
            self.wait_count += 1
            logger.debug(
                "Sem melhora há %d/%d épocas (best=%.6f, atual=%.6f)",
                self.wait_count, self.patience, self.best_loss, val_loss,
            )
            if self.wait_count >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping ativado após %d épocas sem melhora.",
                    self.patience,
                )

        return self.should_stop

    def load_best(self, model: nn.Module) -> None:
        """Carrega o estado do melhor checkpoint no modelo."""
        model.load_state_dict(
            torch.load(self.checkpoint_path, map_location="cpu")
        )
        logger.info("Melhor modelo restaurado de %s", self.checkpoint_path)


# ---------------------------------------------------------------------------
# Funções de treino e validação por época
# ---------------------------------------------------------------------------

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scheduler,               # LambdaLR (step por batch)
    max_norm:  float = 1.0,
) -> float:
    """
    Executa uma época de treinamento com gradient clipping.

    O scheduler é atualizado a cada batch (step-level), que é a prática
    recomendada para schedulers baseados em warmup e cossenoidal
    (Goyal et al., 2017).

    Parâmetros
    ----------
    model     : nn.Module    — modelo em modo treino
    loader    : DataLoader   — DataLoader de treino
    optimizer : Optimizer    — AdamW
    criterion : nn.Module    — função de perda (MSELoss)
    device    : torch.device
    scheduler : LambdaLR     — scheduler step-level
    max_norm  : float        — norma máxima para clipping de gradientes

    Retorna
    -------
    float — loss média da época (em escala normalizada)
    """
    model.train()
    total_loss  = 0.0
    n_batches   = len(loader)

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        # Zera os gradientes acumulados
        optimizer.zero_grad()

        # Forward pass
        pred = model(X)

        # Cálculo da perda (MSE sobre saída normalizada)
        loss = criterion(pred, y)

        # Backward pass
        loss.backward()

        # Clipping de gradientes para estabilidade
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        # Atualização dos parâmetros
        optimizer.step()

        # Atualiza o scheduler a cada batch
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / n_batches


@torch.no_grad()
def validate_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    """
    Avalia o modelo no conjunto de validação.

    Desativa cálculo de gradientes para economizar memória e acelerar
    a inferência.

    Parâmetros
    ----------
    model     : nn.Module
    loader    : DataLoader — DataLoader de validação ou teste
    criterion : nn.Module  — função de perda
    device    : torch.device

    Retorna
    -------
    float — loss média (em escala normalizada)
    """
    model.eval()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred  = model(X)
        loss  = criterion(pred, y)
        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Loop de treinamento completo
# ---------------------------------------------------------------------------

def train_model(
    model:            nn.Module,
    train_loader:     DataLoader,
    val_loader:       DataLoader,
    optimizer:        Optimizer,
    scheduler,        # LambdaLR
    epochs:           int   = 100,
    patience:         int   = 15,
    checkpoint_path:  str   = "checkpoints/best_model.pt",
    max_norm:         float = 1.0,
    device:           torch.device = None,
) -> dict:
    """
    Loop de treinamento completo com Early Stopping.

    Parâmetros
    ----------
    model           : nn.Module
    train_loader    : DataLoader
    val_loader      : DataLoader
    optimizer       : Optimizer  (AdamW pré-configurado)
    scheduler       : LambdaLR   (step-level)
    epochs          : int   — número máximo de épocas
    patience        : int   — patience do Early Stopping
    checkpoint_path : str   — caminho para salvar o melhor modelo
    max_norm        : float — clipping de gradientes
    device          : torch.device

    Retorna
    -------
    dict com:
      'train_losses' : list[float]  — loss de treino por época
      'val_losses'   : list[float]  — loss de validação por época
      'best_val_loss': float        — melhor loss de validação alcançada
      'stopped_epoch': int          — época em que parou
    """
    if device is None:
        device = next(model.parameters()).device

    criterion     = nn.MSELoss()
    early_stop    = EarlyStopping(
        patience        = patience,
        checkpoint_path = checkpoint_path,
    )

    train_losses  = []
    val_losses    = []

    for epoch in range(1, epochs + 1):
        # --- Treino ---
        tr_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device, scheduler, max_norm,
        )

        # --- Validação ---
        va_loss = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        logger.info(
            "Época %3d/%d | Train MSE: %.4f | Val MSE: %.4f | LR: %.2e",
            epoch, epochs, tr_loss, va_loss,
            optimizer.param_groups[0]["lr"],
        )

        # --- Early Stopping ---
        if early_stop(va_loss, model):
            logger.info("Treinamento interrompido na época %d.", epoch)
            break

    # Restaura o melhor modelo
    early_stop.load_best(model)

    return {
        "train_losses":  train_losses,
        "val_losses":    val_losses,
        "best_val_loss": early_stop.best_loss,
        "stopped_epoch": len(train_losses),
    }
