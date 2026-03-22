"""XGBoost stacker fusing snapshot, trajectory, and phenotype outputs."""
import os
from typing import Optional, List

import numpy as np
import xgboost as xgb
import joblib

from models import BaseModel
from utils.logging import get_logger

log = get_logger(__name__)


class FusionModel(BaseModel):
    """Meta-learner: stacks snapshot_prob + trajectory_prob + phenotype_dists."""

    def __init__(self, cfg: dict):
        p = cfg.get("fusion", {})
        self.model = xgb.XGBClassifier(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", 4),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=cfg.get("seed", 42),
            n_jobs=-1,
            tree_method="hist",
        )
        self._cfg = cfg

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        self.model.set_params(scale_pos_weight=n_neg / max(n_pos, 1))

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        log.info(f"FusionModel fitted on meta-features shape {X.shape}")
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        log.info(f"FusionModel saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FusionModel":
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj._cfg = {}
        return obj


def build_meta_features(
    snap_scores: np.ndarray,      # (N,)
    traj_scores: np.ndarray,      # (N,)
    pheno_dists: np.ndarray,      # (N, k)
) -> np.ndarray:
    """Assemble meta-feature matrix: (N, 2+k)."""
    return np.column_stack([
        snap_scores.reshape(-1, 1),
        traj_scores.reshape(-1, 1),
        pheno_dists,
    ])
