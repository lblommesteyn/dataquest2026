"""
RescueWindow — Full Training Pipeline
======================================
Orchestrates: data loading → preprocessing → imputation → feature engineering
→ windowing → splitting → model training (snapshot, trajectory, phenotype, fusion)
→ evaluation → artifact saving.

Usage:
    python train.py

Outputs go to artifacts/models/ and artifacts/results/.
"""
import os
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent))

from config import CFG
from utils.logging import get_logger
from utils.reproducibility import set_all_seeds
from utils.io import save_pickle, save_json

log = get_logger("train")


def main():
    t0 = time.time()
    set_all_seeds(CFG["seed"])

    paths = CFG["paths"]
    models_dir = paths["models_dir"]
    results_dir = paths["results_dir"]
    processed_dir = paths["processed_dir"]
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. LOAD DATA ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 1: Loading raw patient files")
    log.info("=" * 60)

    from data.loader import load_patients, get_raw_dir
    raw_dir = get_raw_dir(CFG)
    patients_raw = load_patients(raw_dir)

    # ── 2. PREPROCESS ────────────────────────────────────────────────────────
    log.info("STEP 2: Preprocessing")
    from data.preprocessor import preprocess_patients
    patients = preprocess_patients(patients_raw, missing_threshold=CFG["missing_threshold"])

    # ── 3. SPLIT PATIENTS ────────────────────────────────────────────────────
    log.info("STEP 3: Patient-level split")
    from data.splitter import split_patients, save_split
    train_ids, val_ids, test_ids = split_patients(
        patients,
        train_frac=CFG["split"]["train"],
        val_frac=CFG["split"]["val"],
        seed=CFG["seed"],
    )
    save_split(train_ids, val_ids, test_ids, os.path.join(processed_dir, "split.json"))

    train_patients = {pid: patients[pid] for pid in train_ids}
    val_patients = {pid: patients[pid] for pid in val_ids}
    test_patients = {pid: patients[pid] for pid in test_ids}

    # ── 4. IMPUTE (fit on train only) ────────────────────────────────────────
    log.info("STEP 4: Imputation")
    from data.loader import FEATURE_COLS
    from data.imputer import PatientImputer

    # Get feature cols present across all data
    all_feat_cols = list(
        set().union(*[set(df.columns) for df in patients.values()])
        - {"SepsisLabel"}
    )
    all_feat_cols = [c for c in FEATURE_COLS if c in all_feat_cols]

    imputer = PatientImputer()
    train_patients_imp = imputer.fit_transform(train_patients, all_feat_cols)
    val_patients_imp = imputer.transform(val_patients)
    test_patients_imp = imputer.transform(test_patients)
    imputer.save(os.path.join(models_dir, "imputer.pkl"))

    # Keep raw (pre-imputation) for missingness indicators
    raw_train = {pid: patients_raw[pid] for pid in train_ids if pid in patients_raw}
    raw_val = {pid: patients_raw[pid] for pid in val_ids if pid in patients_raw}
    raw_test = {pid: patients_raw[pid] for pid in test_ids if pid in patients_raw}

    # ── 5. FEATURE ENGINEERING ───────────────────────────────────────────────
    log.info("STEP 5: Feature engineering")
    from data.feature_engineer import engineer_features, get_feature_cols
    train_eng = engineer_features(train_patients_imp, raw_train)
    val_eng = engineer_features(val_patients_imp, raw_val)
    test_eng = engineer_features(test_patients_imp, raw_test)

    # Determine feature columns from training set
    sample_df = next(iter(train_eng.values()))
    feature_cols = get_feature_cols(sample_df)

    # ── 6. BUILD WINDOWS ─────────────────────────────────────────────────────
    log.info("STEP 6: Building windows")
    from data.windower import build_windows
    horizon = CFG["prediction_horizon_hours"]
    seq_len = CFG["sequence_window_hours"]

    train_windows = build_windows(train_eng, feature_cols, horizon, seq_len)
    val_windows = build_windows(val_eng, feature_cols, horizon, seq_len)
    test_windows = build_windows(test_eng, feature_cols, horizon, seq_len)

    # Save metadata for inference
    save_json(
        {"feature_cols": train_windows["feature_cols"], "traj_cols": train_windows["traj_cols"]},
        os.path.join(models_dir, "feature_meta.json"),
    )

    # ── 7. SNAPSHOT MODEL ────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 7: Training Snapshot Model (XGBoost)")
    log.info("=" * 60)
    from models.snapshot.trainer import train_snapshot
    snap_model = train_snapshot(train_windows, val_windows, CFG)
    snap_model.save(os.path.join(models_dir, "snapshot.pkl"))

    # ── 8. TRAJECTORY MODEL ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 8: Training Trajectory Model (GRU)")
    log.info("=" * 60)
    from models.trajectory.trainer import train_trajectory
    traj_model = train_trajectory(train_windows, val_windows, CFG)
    traj_model.save(os.path.join(models_dir, "trajectory.pt"))

    # ── 9. PHENOTYPE MODEL ───────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 9: Training Phenotype Model (Autoencoder + K-Means)")
    log.info("=" * 60)
    from models.phenotype.trainer import train_phenotype
    pheno_model = train_phenotype(train_windows, CFG)
    pheno_model.save(os.path.join(models_dir, "phenotype"))

    # ── 10. FUSION MODEL ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 10: Training Fusion Model (XGBoost Stacker)")
    log.info("=" * 60)
    from models.fusion.trainer import train_fusion
    fusion_model = train_fusion(snap_model, traj_model, pheno_model, train_windows, val_windows, CFG)
    fusion_model.save(os.path.join(models_dir, "fusion.pkl"))

    # ── 11. EVALUATE ON TEST SET ─────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 11: Evaluation on Test Set")
    log.info("=" * 60)
    from models.fusion.model import build_meta_features
    from evaluation.reporter import evaluate_model, save_results

    y_test = test_windows["y"]
    pids_test = test_windows["patient_ids"]
    hours_test = test_windows["hours"]
    sensitivity_target = CFG["alert_sensitivity_target"]

    snap_scores = snap_model.predict_proba(test_windows["X_snapshot"])
    traj_scores = traj_model.predict_proba(test_windows["X_traj"], mask=test_windows["traj_mask"])
    pheno_dists = pheno_model.predict_proba(test_windows["X_snapshot"])
    X_meta_test = build_meta_features(snap_scores, traj_scores, pheno_dists)
    fusion_scores = fusion_model.predict_proba(X_meta_test)

    # Logistic regression baseline
    log.info("Training logistic regression baseline...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(train_windows["X_snapshot"])
    X_te_scaled = scaler.transform(test_windows["X_snapshot"])
    lr = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=CFG["seed"])
    lr.fit(X_tr_scaled, train_windows["y"])
    lr_scores = lr.predict_proba(X_te_scaled)[:, 1]

    model_results = [
        evaluate_model("logistic_regression", y_test, lr_scores, pids_test, hours_test, sensitivity_target),
        evaluate_model("snapshot_xgb", y_test, snap_scores, pids_test, hours_test, sensitivity_target),
        evaluate_model("trajectory_gru", y_test, traj_scores, pids_test, hours_test, sensitivity_target),
        evaluate_model("fusion_multimodel", y_test, fusion_scores, pids_test, hours_test, sensitivity_target),
    ]

    all_scores = {
        "Logistic Regression": lr_scores,
        "Snapshot XGBoost": snap_scores,
        "Trajectory GRU": traj_scores,
        "Fusion (RescueWindow)": fusion_scores,
    }

    save_results(model_results, all_scores, y_test, results_dir)

    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"Training complete in {elapsed/60:.1f} minutes")
    log.info(f"Models saved to: {models_dir}")
    log.info(f"Results saved to: {results_dir}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
