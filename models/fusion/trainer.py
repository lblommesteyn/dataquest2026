"""Assemble meta-features and train the stacker."""
import numpy as np

from models.fusion.model import FusionModel, build_meta_features
from evaluation.metrics import compute_metrics
from utils.logging import get_logger

log = get_logger(__name__)


def train_fusion(
    snap_model,
    traj_model,
    pheno_model,
    train_data: dict,
    val_data: dict,
    cfg: dict,
) -> FusionModel:
    """
    Train on val-split meta-features (base models were trained on train split,
    so val predictions are out-of-fold and stacking-safe).
    """
    log.info("Building meta-features for fusion stacker...")

    # Val-split meta-features (stacker trains on these)
    snap_val = snap_model.predict_proba(val_data["X_snapshot"])
    traj_val = traj_model.predict_proba(val_data["X_traj"], mask=val_data["traj_mask"])
    pheno_val = pheno_model.predict_proba(val_data["X_snapshot"])
    X_meta_val = build_meta_features(snap_val, traj_val, pheno_val)
    y_val = val_data["y"]

    # Use a portion of meta-val for fusion training and rest for evaluation
    n = len(y_val)
    n_train = int(n * 0.7)
    idx = np.random.default_rng(cfg.get("seed", 42)).permutation(n)
    tr_idx, ev_idx = idx[:n_train], idx[n_train:]

    fusion = FusionModel(cfg)
    fusion.fit(
        X_meta_val[tr_idx], y_val[tr_idx],
        X_val=X_meta_val[ev_idx], y_val=y_val[ev_idx],
    )

    # Log val performance
    val_probs = fusion.predict_proba(X_meta_val)
    metrics = compute_metrics(y_val, val_probs, label="fusion_val")
    log.info(f"Fusion val metrics: {metrics}")

    return fusion
