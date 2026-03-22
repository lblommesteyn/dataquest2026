"""
RescueWindow — Inference Entry Point
======================================
Score a single patient's PSV file and report the rescue window.

Usage:
    python predict.py --patient data/raw/training_setA/p000001.psv
    python predict.py --patient p000001.psv --threshold 0.4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import CFG
from utils.logging import get_logger
from utils.io import load_json

log = get_logger("predict")


def main():
    parser = argparse.ArgumentParser(description="RescueWindow: Score a single patient")
    parser.add_argument("--patient", required=True, help="Path to patient PSV file")
    parser.add_argument(
        "--threshold", type=float,
        default=None,
        help="Alert threshold (default: loaded from training artifacts)"
    )
    parser.add_argument(
        "--models-dir", default=None,
        help="Path to models directory (default: from config)"
    )
    args = parser.parse_args()

    models_dir = args.models_dir or CFG["paths"]["models_dir"]
    threshold = args.threshold or 0.4

    # Load feature metadata
    meta = load_json(str(Path(models_dir) / "feature_meta.json"))
    feature_cols = meta["feature_cols"]
    traj_cols = meta["traj_cols"]

    # Load and preprocess patient
    import pandas as pd
    import numpy as np
    from data.preprocessor import preprocess_patients
    from data.imputer import PatientImputer
    from data.feature_engineer import engineer_features

    patient_path = Path(args.patient)
    if not patient_path.exists():
        log.error(f"Patient file not found: {patient_path}")
        sys.exit(1)

    df_raw = pd.read_csv(patient_path, sep="|")
    pid = patient_path.stem
    raw_patients = {pid: df_raw}

    # Preprocess
    patients = preprocess_patients(raw_patients)

    # Impute using training medians
    imputer = PatientImputer.load(str(Path(models_dir) / "imputer.pkl"))
    patients_imp = imputer.transform(patients)

    # Feature engineering
    patients_eng = engineer_features(patients_imp, raw_patients)
    patient_df = patients_eng[pid]

    # Score
    from inference.scorer import RescueWindowScorer
    from inference.rescue_window import detect_rescue_window

    scorer = RescueWindowScorer(models_dir, alert_threshold=threshold)
    scorer.load_models()

    patient_scores = scorer.score_patient(
        patient_df, feature_cols, traj_cols, seq_len=CFG["sequence_window_hours"]
    )

    # Detect rescue window
    result = detect_rescue_window(
        patient_scores,
        alert_threshold=threshold,
        prediction_horizon_hours=CFG["prediction_horizon_hours"],
    )

    print("\n" + "=" * 60)
    print("RescueWindow — Patient Report")
    print("=" * 60)
    print(result.summary())
    print()
    print("Hour-by-hour risk:")
    print(f"{'Hour':>5} | {'Snapshot':>10} | {'Trajectory':>10} | {'Fusion':>8} | {'Phenotype':<25} | Alert")
    print("-" * 75)
    for hs in patient_scores.hour_scores:
        alert_str = "*** ALERT ***" if hs.alert else ""
        print(
            f"{hs.hour:>5} | {hs.snapshot_risk:>10.3f} | {hs.trajectory_risk:>10.3f} | "
            f"{hs.fusion_risk:>8.3f} | {hs.phenotype_name:<25} | {alert_str}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
