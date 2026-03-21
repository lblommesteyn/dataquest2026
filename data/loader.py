"""Load PhysioNet 2019 Sepsis Challenge data.

Supports two formats:
  - Kaggle CSV format: training_setA.csv / training_setB.csv
    Each row is one patient-hour; a 'PatientID' column identifies patients.
  - PhysioNet PSV format: one .psv file per patient (legacy support).
"""
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from utils.logging import get_logger

log = get_logger(__name__)

# Column definitions (same in both formats)
VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
]
DEMO_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
FEATURE_COLS = VITAL_COLS + LAB_COLS + DEMO_COLS
TARGET_COL = "SepsisLabel"
ALL_COLS = FEATURE_COLS + [TARGET_COL]

# Kaggle dataset column name for patient ID
KAGGLE_PID_COL = "PatientID"


def load_patients(raw_dir: str) -> Dict[str, pd.DataFrame]:
    """Auto-detect format and load all patients.

    Returns dict: patient_id (str) → DataFrame (rows = hours, 0-indexed).
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            "Run: python download_data.py"
        )

    # Try Kaggle CSV format first
    csv_files = sorted(raw_path.glob("training_set*.csv"))
    if csv_files:
        return _load_kaggle_csv(csv_files)

    # Fall back to PSV format (individual files)
    psv_files = sorted(raw_path.glob("*.psv"))
    if psv_files:
        return _load_psv_files(psv_files)

    # Search recursively for PSV or CSV files anywhere under raw_dir
    psv_files = sorted(raw_path.rglob("*.psv"))
    if psv_files:
        return _load_psv_files(psv_files)

    csv_files = sorted(raw_path.rglob("training_set*.csv"))
    if csv_files:
        return _load_kaggle_csv(csv_files)

    raise FileNotFoundError(
        f"No CSV or PSV files found under {raw_dir}.\n"
        "Run: python download_data.py"
    )


def _load_kaggle_csv(csv_files: List[Path]) -> Dict[str, pd.DataFrame]:
    """Load combined CSV files (Kaggle format)."""
    log.info(f"Loading Kaggle CSV format: {[f.name for f in csv_files]}")
    frames = []
    for fpath in csv_files:
        df = pd.read_csv(fpath)
        source = fpath.stem  # e.g. "training_setA"
        df["_source"] = source
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Loaded {len(combined):,} rows, columns: {list(combined.columns[:8])}...")

    # Determine patient ID column
    pid_col = None
    for candidate in [KAGGLE_PID_COL, "Patient_ID", "patient_id", "ID"]:
        if candidate in combined.columns:
            pid_col = candidate
            break

    if pid_col is None:
        raise ValueError(
            f"Cannot find patient ID column. Available columns: {list(combined.columns)}"
        )

    patients = {}
    for pid, group in tqdm(combined.groupby(pid_col), desc="Splitting by patient"):
        df = group.drop(columns=[pid_col, "_source"], errors="ignore")
        df = df.reset_index(drop=True)
        patients[str(pid)] = df

    log.info(f"Loaded {len(patients):,} patients")
    return patients


def _load_psv_files(psv_files: List[Path]) -> Dict[str, pd.DataFrame]:
    """Load individual PSV files (PhysioNet format)."""
    log.info(f"Loading PSV format: {len(psv_files)} files")
    patients = {}
    for fpath in tqdm(psv_files, desc="Loading patients"):
        pid = fpath.stem
        try:
            df = pd.read_csv(fpath, sep="|")
            df = df.reset_index(drop=True)
            patients[pid] = df
        except Exception as e:
            log.warning(f"Skipping {fpath.name}: {e}")
    log.info(f"Loaded {len(patients):,} patients")
    return patients


def get_raw_dir(cfg: dict) -> str:
    return cfg["paths"]["raw_dir"]
# iteration 3: loader updated
# iteration 13: loader updated
# iteration 23: loader updated
# iteration 33: loader updated
