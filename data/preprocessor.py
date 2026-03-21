"""Clip physiological values to plausible ranges, encode categoricals."""
from typing import Dict

import numpy as np
import pandas as pd

from data.loader import VITAL_COLS, LAB_COLS, DEMO_COLS, FEATURE_COLS, TARGET_COL
from utils.logging import get_logger

log = get_logger(__name__)

# Clinical plausibility bounds (lo, hi) — values outside are set to NaN
CLIP_BOUNDS = {
    "HR": (0, 300),
    "O2Sat": (0, 100),
    "Temp": (25, 45),
    "SBP": (30, 300),
    "MAP": (20, 200),
    "DBP": (10, 200),
    "Resp": (0, 80),
    "EtCO2": (0, 80),
    "pH": (6.5, 8.0),
    "FiO2": (0.21, 1.0),
    "Creatinine": (0, 30),
    "WBC": (0, 200),
    "Platelets": (0, 2000),
    "Glucose": (10, 1500),
    "Lactate": (0, 30),
    "Age": (0, 120),
    "ICULOS": (0, 500),
}


def preprocess_patients(
    patients: Dict[str, pd.DataFrame],
    missing_threshold: float = 0.80,
) -> Dict[str, pd.DataFrame]:
    """Apply clipping, encoding, and missingness filtering per patient."""
    out = {}
    drop_stats = {"too_short": 0, "processed": 0}

    for pid, df in patients.items():
        df = df.copy()

        # Encode Gender: M=1, F=0, unknown=NaN
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].map({1: 1, 0: 0, "M": 1, "F": 0})

        # Unit columns: already binary 0/1, fill missing with 0
        for col in ["Unit1", "Unit2"]:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(float)

        # Clip physiological values
        for col, (lo, hi) in CLIP_BOUNDS.items():
            if col in df.columns:
                df[col] = df[col].where(
                    (df[col] >= lo) & (df[col] <= hi), other=np.nan
                )

        # Drop features where this patient is missing > threshold fraction
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        miss_frac = df[feature_cols].isna().mean()
        keep_cols = miss_frac[miss_frac <= missing_threshold].index.tolist()
        keep_cols = keep_cols + [TARGET_COL] if TARGET_COL in df.columns else keep_cols
        df = df[keep_cols]

        out[pid] = df
        drop_stats["processed"] += 1

    log.info(
        f"Preprocessed {drop_stats['processed']} patients "
        f"(dropped {drop_stats['too_short']} too short)"
    )
    return out
# iteration 4: preprocessor updated
# iteration 14: preprocessor updated
# iteration 24: preprocessor updated
# iteration 34: preprocessor updated
