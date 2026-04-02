"""
main.py
=======
Script principal que orquestra o pipeline completo de previsão
meteorológica com o WeatherTransformer.

Execução:
    python main.py [--skip-tuning] [--n-trials N] [--device cuda]

Argumentos opcionais:
    --skip-tuning : pula a otimização Optuna e usa hiperparâmetros padrão
    --n-trials N  : número de trials Optuna (default: 100)
    --device      : 'cuda', 'mps' ou 'cpu' (auto-detectado se omitido)

Pipeline completo (13 etapas):
    1.  Leitura dos dados NetCDF
    2.  Merging das variáveis ERA5-Land
    3.  Engenharia de features físicas
    4.  Normalização por tipo de feature
    5.  Criação das janelas temporais (sliding window)
    6.  Split train / validation / test (temporal)
    7.  Temporal Patch Embedding
    8.  Ajuste estrutural via Optuna (d_model, layers, heads)
    9.  Temporal Decay Bias (ALiBi)
    10. Regularização (dropout + early stopping)
    11. Learning rate scheduler (cosine / cosine+warmup)
    12. Avaliação no conjunto de teste
    13. Explicabilidade (atenção + saliência)
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch

import config as cfg
from data.loader      import load_and_merge_nc
from data.features    import build_features
from data.normalization import FeatureNormalizer
from data.dataset     import temporal_split, create_sequences, build_dataloaders
from model.transformer import WeatherTransformer
from model.scheduler   import build_scheduler
from training.trainer  import train_model
from training.evaluate import (
    evaluate_on_test, predict,
    plot_learning_curves, plot_predictions,
)
from tuning.optuna_search import run_optuna_study, best_hyperparams_from_study
from explainability.explain import explain_sample

# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------
# O FileHandler aponta para o diretório timestampado desta execução.
# cfg.OUTPUT_DIR já contém o timestamp (gerado em config.py na importação),
# portanto basta usá-lo diretamente — o diretório foi criado em config.py.

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(cfg.OUTPUT_DIR, "run.log")),
    ],
)
logger = logging.getLogger("main")
logger.info("Run ID (timestamp): %s", cfg.RUN_TIMESTAMP)
logger.info("Checkpoints  -> %s", cfg.CHECKPOINT_DIR)
logger.info("Outputs      -> %s", cfg.OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------

def set_seed(seed: int = cfg.SEED) -> None:
    """Fixa a semente aleatória em todas as bibliotecas relevantes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Para uso com PyTorch >= 1.8
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Detecção do dispositivo de computação
# ---------------------------------------------------------------------------

def get_device(device_str: str | None = None) -> torch.device:
    """
    Seleciona automaticamente o dispositivo (CUDA > MPS > CPU) ou
    usa o explicitamente fornecido.
    """
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:

    set_seed(cfg.SEED)
    device = get_device(args.device)
    logger.info("Dispositivo: %s", device)

    # ====================================================================
    # ETAPAS 1 e 2: Leitura e Merging
    # ====================================================================
    logger.info("=== ETAPA 1-2: Leitura e Merging dos dados ERA5-Land ===")
    df = load_and_merge_nc(cfg.NC_FILES)

    # ====================================================================
    # ETAPA 3: Engenharia de Features Físicas e Temporais
    # ====================================================================
    logger.info("=== ETAPA 3: Feature Engineering ===")
    df = build_features(df)

    # ====================================================================
    # ETAPA 6: Split temporal (antes de normalizar para evitar leakage)
    # ====================================================================
    logger.info("=== ETAPA 6: Split Temporal ===")
    train_df, val_df, test_df = temporal_split(
        df, train_end=cfg.TRAIN_END, val_end=cfg.VAL_END
    )

    # ====================================================================
    # ETAPA 4: Normalização (ajuste APENAS no treino)
    # ====================================================================
    logger.info("=== ETAPA 4: Normalização ===")
    normalizer = FeatureNormalizer(
        physical_cols = cfg.PHYSICAL_FEATURES,
        cyclic_cols   = cfg.CYCLIC_FEATURES,
        target_col    = cfg.TARGET,
    )
    normalizer.fit(train_df)

    # Transforma features e target de cada partição
    X_train = normalizer.transform_features(train_df)
    y_train = normalizer.transform_target(train_df)

    X_val   = normalizer.transform_features(val_df)
    y_val   = normalizer.transform_target(val_df)

    X_test  = normalizer.transform_features(test_df)
    y_test  = normalizer.transform_target(test_df)

    # Salva o normalizer para uso em inferência futura
    normalizer.save(os.path.join(cfg.CHECKPOINT_DIR, "normalizer.pkl"))

    # ====================================================================
    # ETAPA 5: Criação das Janelas Temporais
    # ====================================================================
    logger.info("=== ETAPA 5: Janelas Temporais (window=%d, horizon=%d) ===",
                cfg.WINDOW, cfg.HORIZON)
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, cfg.WINDOW, cfg.HORIZON)
    X_val_seq,   y_val_seq   = create_sequences(X_val,   y_val,   cfg.WINDOW, cfg.HORIZON)
    X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  cfg.WINDOW, cfg.HORIZON)

    logger.info(
        "Shapes: treino=%s | val=%s | teste=%s",
        X_train_seq.shape, X_val_seq.shape, X_test_seq.shape,
    )

    # ====================================================================
    # ETAPAS 7–11: Optuna (ou hiperparâmetros padrão)
    # ====================================================================
    n_trials = args.n_trials

    if args.skip_tuning:
        logger.info("=== Pulando Optuna. Usando hiperparâmetros padrão. ===")
        best_params = {
            "n_layers":     4,
            "d_model":      256,
            "n_heads":      4,
            "dim_ff":       1024,
            "attn_dropout": 0.1,
            "ff_dropout":   0.1,
            "learning_rate": 1e-3,
            "weight_decay":  1e-4,
            "batch_size":    128,
            "scheduler":     "cosine_warmup",
            "warmup_steps":  500,
        }
    else:
        logger.info("=== ETAPAS 7-11: Otimização de Hiperparâmetros (Optuna, %d trials) ===",
                    n_trials)

        # DataLoaders para o Optuna (usa batch_size padrão; cada trial
        # reamostra batch_size mas usa o mesmo loader — simplificação
        # válida pois o impacto de batch_size é capturado pela LR ajustada)
        _train_ldr, _val_ldr, _ = build_dataloaders(
            X_train_seq, y_train_seq,
            X_val_seq,   y_val_seq,
            X_test_seq,  y_test_seq,
            batch_size = cfg.DEFAULT_BATCH_SIZE,
        )

        study = run_optuna_study(
            train_loader = _train_ldr,
            val_loader   = _val_ldr,
            device       = device,
            n_trials     = n_trials,
            storage      = "sqlite:///" + os.path.join(cfg.OUTPUT_DIR, "optuna.db"),
        )

        best_params = best_hyperparams_from_study(study)
        logger.info("Melhores hiperparâmetros: %s", best_params)

    # ====================================================================
    # Treinamento final com os melhores hiperparâmetros
    # ====================================================================
    logger.info("=== Treinamento Final ===")

    batch_size = best_params.get("batch_size", cfg.DEFAULT_BATCH_SIZE)
    train_loader, val_loader, test_loader = build_dataloaders(
        X_train_seq, y_train_seq,
        X_val_seq,   y_val_seq,
        X_test_seq,  y_test_seq,
        batch_size = batch_size,
    )

    # Constrói o modelo com os melhores hiperparâmetros
    # (Etapas 7, 8 e 9 — embedding, estrutura e decay bias)
    model = WeatherTransformer(
        input_dim    = cfg.INPUT_DIM,
        d_model      = best_params.get("d_model",      256),
        n_heads      = best_params.get("n_heads",       4),
        n_layers     = best_params.get("n_layers",      4),
        dim_ff       = best_params.get("dim_ff",      1024),
        attn_dropout = best_params.get("attn_dropout", 0.1),
        ff_dropout   = best_params.get("ff_dropout",   0.1),
        horizon      = cfg.HORIZON,
        seq_len      = cfg.WINDOW,
        patch_size   = cfg.PATCH_SIZE,
        use_alibi    = True,    # Temporal Decay Bias (Etapa 9)
    ).to(device)

    logger.info("Parâmetros treináveis: %d", model.count_parameters())

    # Otimizador AdamW (com weight decay desacoplado)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = best_params.get("learning_rate", cfg.DEFAULT_LR),
        weight_decay = best_params.get("weight_decay",  cfg.DEFAULT_WEIGHT_DECAY),
    )

    # Scheduler de LR (Etapa 11)
    total_steps  = cfg.DEFAULT_EPOCHS * len(train_loader)
    scheduler = build_scheduler(
        optimizer      = optimizer,
        scheduler_type = best_params.get("scheduler", "cosine_warmup"),
        total_steps    = total_steps,
        warmup_steps   = best_params.get("warmup_steps", 500),
    )

    # Treinamento com Early Stopping (Etapa 10)
    history = train_model(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        optimizer       = optimizer,
        scheduler       = scheduler,
        epochs          = cfg.DEFAULT_EPOCHS,
        patience        = cfg.DEFAULT_PATIENCE,
        checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt"),
        device          = device,
    )

    # Curvas de aprendizado
    plot_learning_curves(
        history["train_losses"],
        history["val_losses"],
        save_path=os.path.join(cfg.OUTPUT_DIR, "learning_curves.pdf"),
    )

    # ====================================================================
    # ETAPA 12: Avaliação no Conjunto de Teste
    # ====================================================================
    logger.info("=== ETAPA 12: Avaliação no Conjunto de Teste ===")
    metrics = evaluate_on_test(model, test_loader, normalizer, device)

    # Gráfico de previsões
    y_true, y_pred = predict(model, test_loader, normalizer, device)
    plot_predictions(
        y_true, y_pred, n_samples=3,
        save_path=os.path.join(cfg.OUTPUT_DIR, "predictions.pdf"),
    )

    # ====================================================================
    # ETAPA 13: Explicabilidade
    # ====================================================================
    logger.info("=== ETAPA 13: Explicabilidade ===")
    # Usa a primeira amostra do conjunto de teste como exemplo
    sample_x = torch.tensor(X_test_seq[0], dtype=torch.float32)
    explain_sample(
        model         = model,
        x_sample      = sample_x,
        feature_names = cfg.FEATURES,
        device        = device,
        output_dir    = cfg.OUTPUT_DIR,
    )

    # ====================================================================
    # Relatório final
    # ====================================================================
    logger.info("=" * 60)
    logger.info("PIPELINE CONCLUIDO")
    logger.info("  Run ID      : %s", cfg.RUN_TIMESTAMP)
    logger.info("  RMSE (teste): %.4f graus C", metrics["rmse"])
    logger.info("  MAE  (teste): %.4f graus C", metrics["mae"])
    logger.info("  Figuras     : %s/", cfg.OUTPUT_DIR)
    logger.info("  Checkpoints : %s/", cfg.CHECKPOINT_DIR)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WeatherTransformer — Previsão de Temperatura em Campinas"
    )
    parser.add_argument(
        "--skip-tuning", action="store_true",
        help="Pula a otimização Optuna e usa hiperparâmetros padrão.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=cfg.N_TRIALS,
        help=f"Número de trials Optuna (default: {cfg.N_TRIALS}).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Dispositivo: 'cuda', 'mps' ou 'cpu' (auto se omitido).",
    )

    args = parser.parse_args()
    main(args)
