"""Patient-level train/val/test split (no row leakage)."""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.loader import TARGET_COL
from utils.logging import get_logger

log = get_logger(__name__)


def _patient_has_sepsis(df: pd.DataFrame) -> int:
    """1 if patient ever develops sepsis, else 0."""
    if TARGET_COL not in df.columns:
        return 0
    return int(df[TARGET_COL].max() > 0)


def split_patients(
    patients: Dict[str, pd.DataFrame],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Stratified patient-level split.

    Returns (train_ids, val_ids, test_ids).
    """
    pids = list(patients.keys())
    labels = [_patient_has_sepsis(patients[p]) for p in pids]

    test_frac = 1.0 - train_frac - val_frac

    train_ids, temp_ids, _, temp_labels = train_test_split(
        pids, labels, test_size=(1 - train_frac), stratify=labels, random_state=seed
    )
    relative_val = val_frac / (val_frac + test_frac)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=(1 - relative_val), stratify=temp_labels, random_state=seed
    )

    log.info(
        f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
    )
    return train_ids, val_ids, test_ids


def apply_split(
    windows: Dict,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> Tuple[Dict, Dict, Dict]:
    """Filter window arrays by patient split."""
    pid_array = windows["patient_ids"]

    def _mask(ids):
        id_set = set(ids)
        return np.array([p in id_set for p in pid_array])

    def _filter(m):
        return {
            "X_snapshot": windows["X_snapshot"][m],
            "X_traj": windows["X_traj"][m],
            "traj_mask": windows["traj_mask"][m],
            "y": windows["y"][m],
            "patient_ids": windows["patient_ids"][m],
            "hours": windows["hours"][m],
            "feature_cols": windows["feature_cols"],
            "traj_cols": windows["traj_cols"],
        }

    train_m = _mask(train_ids)
    val_m = _mask(val_ids)
    test_m = _mask(test_ids)

    log.info(
        f"Split sizes → train: {train_m.sum():,}, "
        f"val: {val_m.sum():,}, test: {test_m.sum():,}"
    )
    return _filter(train_m), _filter(val_m), _filter(test_m)


def save_split(train_ids, val_ids, test_ids, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f)


def load_split(path: str) -> Tuple[List[str], List[str], List[str]]:
    with open(path) as f:
        d = json.load(f)
    return d["train"], d["val"], d["test"]
