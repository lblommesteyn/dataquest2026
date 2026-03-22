# RescueWindow
### Early Sepsis Deterioration Forecasting with Multimodel Patient Trajectories

> **Core question**: Can we identify a reversible deterioration window *before* sepsis fully declares itself, using only scratch-trained models on hourly patient trajectories?

---

## What it builds

A multimodel early warning engine with four learned components:

| Component | Model | Input | Purpose |
|---|---|---|---|
| Snapshot specialist | XGBoost | Current-hour features + rolling stats | Fast, interpretable baseline |
| Trajectory specialist | PyTorch GRU | Last 12 hours of raw signals | Captures silent temporal drift |
| Phenotype model | Autoencoder + K-Means | Snapshot embedding | Learns 8 clinical archetypes |
| Fusion model | XGBoost stacker | All three outputs | Final calibrated risk score |

For each patient-hour, the system produces:
- Current sepsis risk (0–1)
- Which model is most concerned
- Patient archetype (e.g., "early_inflammatory", "rapid_collapse")
- **Estimated rescue window** — how many hours remain before projected onset

---

## Dataset

**PhysioNet 2019 Sepsis Challenge (Kaggle mirror)** — 40,336 patients, hourly time-series, ~40 vitals/labs.

- Kaggle dataset: https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis
- Format: `training_setA.csv` + `training_setB.csv`, each row = one patient-hour with a `PatientID` column.

### Download

```bash
# One-time Kaggle API setup:
# Go to https://www.kaggle.com/settings → Create New Token → save to ~/.kaggle/kaggle.json

python download_data.py
```

Files land in `data/raw/training_setA.csv` and `data/raw/training_setB.csv`.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch. GPU optional but recommended for GRU training.

---

## Train

```bash
cd rescuewindow/
python train.py
```

This runs the full pipeline:
1. Load raw PSV files
2. Preprocess + clip physiological values
3. Patient-level 70/15/15 split (no row leakage)
4. Forward-fill + population median imputation (fit on train only)
5. Feature engineering: rolling means/slopes/stds, derived vitals, missingness flags
6. Build snapshot arrays `(N, ~250)` and trajectory tensors `(N, 12, 40)`
7. Train Snapshot XGBoost
8. Train Trajectory GRU (PyTorch)
9. Train Phenotype Autoencoder + K-Means
10. Train Fusion XGBoost stacker
11. Evaluate all 4 models vs logistic regression baseline
12. Save comparison table, ROC/PR curves, and all model artifacts

### Outputs

```
artifacts/
├── models/
│   ├── snapshot.pkl        ← XGBoost snapshot model
│   ├── trajectory.pt       ← GRU weights
│   ├── phenotype_ae.pt     ← Autoencoder weights
│   ├── phenotype_clusterer.pkl
│   ├── fusion.pkl          ← Stacker
│   ├── imputer.pkl         ← Training-set medians
│   └── feature_meta.json   ← Feature column list
└── results/
    ├── comparison_table.csv  ← AUROC/AUPRC/F1/lead-time per model
    ├── curves.png            ← ROC + PR curves
    └── results.json          ← Full metrics
```

---

## Predict on a new patient

```bash
python predict.py --patient data/raw/training_setA/p000001.psv
```

Prints an hour-by-hour risk table and a rescue window report:

```
==============================
RescueWindow — Patient Report
==============================
[p000001] ALERT at hour 14 | Predicted onset: hour 17 | Rescue window: 4.0h | Confidence: 0.73 | Phenotype: early_inflammatory | Trend: rising

Hour | Snapshot   | Trajectory | Fusion   | Phenotype                  | Alert
----------------------------------------------------------------------
   0 |      0.021 |      0.018 |    0.019 | stable                     |
   1 |      0.024 |      0.022 |    0.020 | stable                     |
  ...
  14 |      0.681 |      0.720 |    0.731 | early_inflammatory         | *** ALERT ***
```

---

## Evaluation metrics

| Metric | Description |
|---|---|
| AUROC | Area under ROC curve |
| AUPRC | Area under precision-recall curve |
| F1 @ {0.3, 0.4, 0.5} | F1 at operating thresholds |
| Sensitivity @ Specificity 90% | Clinical operating point |
| Detection rate | % of sepsis patients with any alert before onset |
| Mean lead time (h) | Average hours of warning before onset |
| % early by 6h | % of caught patients alerted ≥6h before onset |

---

## Project structure

```
rescuewindow/
├── config/            YAML configs + loader
├── data/              Load → preprocess → impute → feature engineer → window → split
├── models/
│   ├── snapshot/      XGBoost snapshot model
│   ├── trajectory/    GRU trajectory model
│   ├── phenotype/     Autoencoder + k-means
│   └── fusion/        XGBoost stacker
├── evaluation/        Metrics, lead-time, comparison reporter
├── inference/         Per-patient scorer + rescue window detector
├── utils/             Logging, IO, seeds
├── train.py           Full training pipeline
├── predict.py         Single-patient inference
└── download_data.py   PhysioNet data downloader
```

---

## Key design decisions

- **Patient-level split**: No patient appears in both train and test. Row-level splits cause severe leakage.
- **Imputation discipline**: Population medians are computed on train only, then applied to val/test.
- **Target**: "Sepsis onset within next 6 hours" — not just current label — gives the model a real predictive task.
- **Stacking safety**: Base models train on train split; fusion stacker trains on val split predictions (out-of-fold).
- **Phenotype as soft features**: Distances to all k=8 centroids are passed to the stacker (not just the hard cluster label), giving it richer signal.
- **Rescue window**: Defined as `prediction_horizon - clinical_action_lag` from the first alert, not from sepsis onset — reflecting real clinical utility.
