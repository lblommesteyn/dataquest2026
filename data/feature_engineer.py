"""Rolling stats, derived features, missingness indicators."""
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import linregress

from data.loader import VITAL_COLS, LAB_COLS, FEATURE_COLS, TARGET_COL
from utils.logging import get_logger

log = get_logger(__name__)

ROLLING_WINDOWS = [3, 6, 12]  # hours
ROLLING_FEATURES = VITAL_COLS + ["pH", "Lactate", "Creatinine", "WBC", "Platelets"]


def _slope(x: np.ndarray) -> float:
    valid = ~np.isnan(x)
    if valid.sum() < 2:
        return np.nan
    t = np.arange(len(x))[valid]
    y = x[valid]
    try:
        slope, _, _, _, _ = linregress(t, y)
        return slope
    except Exception:
        return np.nan


def _compute_rolling_cols(df: pd.DataFrame, col: str, window: int) -> Dict[str, list]:
    series = df[col].values.astype(float)
    n = len(series)
    means, stds, slopes = [], [], []
    for i in range(n):
        start = max(0, i - window + 1)
        w = series[start : i + 1]
        means.append(np.nanmean(w))
        stds.append(np.nanstd(w))
        slopes.append(_slope(w))
    suffix = f"_w{window}"
    return {
        f"{col}_mean{suffix}": means,
        f"{col}_std{suffix}": stds,
        f"{col}_slope{suffix}": slopes,
    }


def engineer_features(
    patients: Dict[str, pd.DataFrame],
    raw_patients: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Apply all feature engineering to each patient.

    Builds all new columns in one pd.concat call per patient to avoid
    DataFrame fragmentation warnings.
    """
    out = {}

    for pid, df in patients.items():
        base = df.copy()
        raw = raw_patients.get(pid, df)
        new_cols: Dict[str, list] = {}

        # Missingness indicators (from raw, pre-imputation)
        for col in VITAL_COLS + LAB_COLS:
            if col in raw.columns:
                new_cols[f"obs_{col}"] = (~raw[col].isna()).astype(float).tolist()

        # Rolling statistics
        for col in ROLLING_FEATURES:
            if col not in base.columns:
                continue
            for w in ROLLING_WINDOWS:
                new_cols.update(_compute_rolling_cols(base, col, w))

        # Derived clinical features
        if "HR" in base.columns and "SBP" in base.columns:
            new_cols["shock_index"] = (base["HR"] / base["SBP"].replace(0, np.nan)).tolist()
        if "SBP" in base.columns and "DBP" in base.columns:
            new_cols["pulse_pressure"] = (base["SBP"] - base["DBP"]).tolist()
        if "BUN" in base.columns and "Creatinine" in base.columns:
            new_cols["bun_creatinine_ratio"] = (base["BUN"] / base["Creatinine"].replace(0, np.nan)).tolist()
        if "O2Sat" in base.columns and "FiO2" in base.columns:
            new_cols["spo2_fio2"] = (base["O2Sat"] / base["FiO2"].replace(0, np.nan)).tolist()
        if "Resp" in base.columns and "MAP" in base.columns:
            new_cols["resp_map_ratio"] = (base["Resp"] / base["MAP"].replace(0, np.nan)).tolist()

        # Join all new columns at once — no fragmentation
        extra = pd.DataFrame(new_cols, index=base.index)
        out[pid] = pd.concat([base, extra], axis=1)

    log.info(
        f"Feature engineering done. Example feature count: "
        f"{len(out[next(iter(out))].columns) if out else 0}"
    )
    return out


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {TARGET_COL, "hour", "patient_id"}
    return [c for c in df.columns if c not in exclude]
# iteration 1: tuning complete
# iteration 11: tuning complete
# iteration 21: tuning complete
# iteration 31: tuning complete
# iteration 41: tuning complete
