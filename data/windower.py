"""Convert per-patient DataFrames into labeled training examples.

Snapshot: one row per patient-hour → (N, F)
Trajectory: 12-hour window per patient-hour → (N, 12, F_base)
Label: 1 if sepsis onset within next `horizon` hours, else 0
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.loader import VITAL_COLS, LAB_COLS, TARGET_COL
from data.feature_engineer import get_feature_cols
from utils.logging import get_logger

log = get_logger(__name__)

# Base features used in trajectory tensor (raw signals only, no rolling)
TRAJ_BASE_COLS = VITAL_COLS + [
    "pH", "Lactate", "Creatinine", "WBC", "Platelets", "HCO3",
    "Glucose", "Potassium", "Calcium", "Temp",
]


def _build_label(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """For each hour, 1 if sepsis occurs within the next `horizon` hours."""
    label = df[TARGET_COL].values.astype(float)
    n = len(label)
    y = np.zeros(n, dtype=np.int32)
    for i in range(n):
        window = label[i : i + horizon + 1]  # include current + next horizon hours
        if window.max() > 0:
            y[i] = 1
    return y


def _extract_trajectory(
    df: pd.DataFrame, traj_cols: List[str], seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (N, seq_len, len(traj_cols)) tensor and mask (N, seq_len).

    Always produces exactly len(traj_cols) features; missing columns → 0.
    """
    n = len(df)
    n_feats = len(traj_cols)
    # Build aligned data matrix: (n, n_feats), missing cols stay 0
    data = np.zeros((n, n_feats), dtype=np.float32)
    for j, col in enumerate(traj_cols):
        if col in df.columns:
            data[:, j] = df[col].values.astype(np.float32)

    trajectories = np.zeros((n, seq_len, n_feats), dtype=np.float32)
    masks = np.zeros((n, seq_len), dtype=np.float32)

    for i in range(n):
        start = max(0, i - seq_len + 1)
        chunk = data[start : i + 1]
        actual_len = len(chunk)
        trajectories[i, seq_len - actual_len :] = chunk
        masks[i, seq_len - actual_len :] = 1.0

    return trajectories, masks


def build_windows(
    patients: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    horizon: int = 6,
    seq_len: int = 12,
) -> Dict:
    """Build all training arrays from engineered patient DataFrames.

    Returns dict with keys:
        X_snapshot: (N, n_features)
        X_traj:     (N, seq_len, n_traj_features)
        traj_mask:  (N, seq_len)
        y:          (N,)
        patient_ids: (N,) str array
        hours:       (N,) int array
        feature_cols: list of snapshot feature names
        traj_cols:    list of trajectory feature names
    """
    all_snapshot = []
    all_traj = []
    all_mask = []
    all_y = []
    all_pid = []
    all_hour = []

    # Fix trajectory cols to the full TRAJ_BASE_COLS list upfront so every
    # patient produces the same tensor width (missing cols filled with 0 in
    # _extract_trajectory).
    traj_cols_present = TRAJ_BASE_COLS

    for pid, df in tqdm(patients.items(), desc="Building windows"):
        if TARGET_COL not in df.columns:
            continue

        # Snapshot — always produce (n_hours, len(feature_cols)); missing cols → 0
        snap = np.zeros((len(df), len(feature_cols)), dtype=np.float32)
        for j, col in enumerate(feature_cols):
            if col in df.columns:
                snap[:, j] = df[col].values.astype(np.float32)

        # Trajectory
        traj, mask = _extract_trajectory(df, traj_cols_present, seq_len)

        # Labels
        y = _build_label(df, horizon)

        n = len(snap)
        all_snapshot.append(snap)
        all_traj.append(traj)
        all_mask.append(mask)
        all_y.append(y)
        all_pid.extend([pid] * n)
        all_hour.extend(list(range(n)))

    X_snapshot = np.concatenate(all_snapshot, axis=0)
    X_traj = np.concatenate(all_traj, axis=0)
    traj_mask = np.concatenate(all_mask, axis=0)
    y = np.concatenate(all_y, axis=0)
    patient_ids = np.array(all_pid)
    hours = np.array(all_hour)

    # Align feature_cols to actual columns in snapshot
    feat_cols_aligned = [c for c in feature_cols if c in (traj_cols_present or [])]
    # Just use the ones that made it into X_snapshot
    feat_cols_used = [c for c in feature_cols if c in patients[next(iter(patients))].columns]

    pos_rate = y.mean()
    log.info(
        f"Windows built: {len(y):,} examples, "
        f"{int(y.sum()):,} positive ({pos_rate:.2%}), "
        f"snapshot shape {X_snapshot.shape}, "
        f"trajectory shape {X_traj.shape}"
    )

    return {
        "X_snapshot": X_snapshot,
        "X_traj": X_traj,
        "traj_mask": traj_mask,
        "y": y,
        "patient_ids": patient_ids,
        "hours": hours,
        "feature_cols": feat_cols_used,
        "traj_cols": traj_cols_present,
    }
# iteration 5: windower validated
# iteration 15: windower validated
# iteration 25: windower validated
# iteration 35: windower validated
