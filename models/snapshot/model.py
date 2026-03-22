"""XGBoost snapshot specialist — current-hour features only."""
import os
from typing import List, Optional

import numpy as np
import xgboost as xgb
import joblib

from models import BaseModel
from utils.logging import get_logger

log = get_logger(__name__)


class SnapshotModel(BaseModel):
    """XGBoost trained on the full snapshot feature vector."""

    def __init__(self, cfg: dict):
        p = cfg.get("snapshot", {})
        self.model = xgb.XGBClassifier(
            n_estimators=p.get("n_estimators", 500),
            max_depth=p.get("max_depth", 6),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=cfg.get("seed", 42),
            n_jobs=-1,
            tree_method="hist",
        )
        self.early_stopping_rounds = p.get("early_stopping_rounds", 50)
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X, y, X_val=None, y_val=None, feature_names=None, **kwargs):
        self.feature_names_ = feature_names
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        self.model.set_params(scale_pos_weight=n_neg / max(n_pos, 1))

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )
        log.info("SnapshotModel fitted")
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self) -> Optional[np.ndarray]:
        return self.model.feature_importances_

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names_}, path)
        log.info(f"SnapshotModel saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SnapshotModel":
        state = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = state["model"]
        obj.feature_names_ = state["feature_names"]
        obj.early_stopping_rounds = 50
        return obj
