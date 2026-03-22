"""Train the snapshot XGBoost model and return scores."""
import numpy as np

from models.snapshot.model import SnapshotModel
from evaluation.metrics import compute_metrics
from utils.logging import get_logger

log = get_logger(__name__)


def train_snapshot(train_data: dict, val_data: dict, cfg: dict) -> SnapshotModel:
    X_tr, y_tr = train_data["X_snapshot"], train_data["y"]
    X_val, y_val = val_data["X_snapshot"], val_data["y"]
    feature_cols = train_data["feature_cols"]

    model = SnapshotModel(cfg)
    model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, feature_names=feature_cols)

    val_scores = model.predict_proba(X_val)
    metrics = compute_metrics(y_val, val_scores, label="snapshot_val")
    log.info(f"Snapshot val metrics: {metrics}")

    return model
