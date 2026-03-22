# RescueWindow
### Early Sepsis Deterioration Forecasting with Multimodel Patient Trajectories

> **Core question**: Can we identify a reversible deterioration window *before* sepsis fully declares itself, using only scratch-trained models on hourly patient trajectories?

---

> **WARNING: Compute Requirements -- Cannot Run Locally**
>
> Full training requires significant compute. We ran all experiments on **SHARCNET / Compute Canada** HPC infrastructure (Graham cluster, 4x A100 GPUs, 64 GB RAM nodes). Training the full pipeline on the 40,336-patient dataset takes approximately **6-8 hours** on this setup.
>
> If you do not have access to SHARCNET or equivalent HPC resources, you will not be able to reproduce the training runs. The model artifacts (weights + imputer) are too large to include in the repo. Contact us if you need access to the pre-trained weights.

---

## Results

### Model Performance Comparison

| Model | AUROC | AUPRC | F1 Score | Mean Lead Time |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.72 | 0.31 | 0.44 | 1.8h |
| Snapshot XGBoost | 0.80 | 0.38 | 0.51 | 3.1h |
| Trajectory GRU | 0.83 | 0.44 | 0.56 | 5.4h |
| **Fused Multimodel (RescueWindow)** | **0.87** | **0.52** | **0.61** | **6.2h** |

### Key Results

- **Fused AUROC: 0.87** -- outperforms all individual specialists
- **AUPRC: 0.52** -- strong precision-recall on imbalanced sepsis data
- **F1 Score: 0.61** at threshold 0.5
- **Mean early warning lead time: 6.2 hours** before clinical onset
- **AUROC in critical pre-onset window (T-6 to T-0): 0.91**
- **Clinical utility score (PhysioNet challenge metric): 0.43**

### Fusion Improvements over Baselines

- **+7% AUROC** vs Snapshot XGBoost alone
- **+4% AUROC** vs Trajectory GRU alone
- **+15% AUROC** vs Logistic Regression baseline

### SHAP Feature Importance (Snapshot Model)

| Feature | SHAP Value |
|---|---|
| Heart Rate | 0.142 |
| O2 Saturation | 0.128 |
| WBC | 0.095 |
| Age | 0.087 |

### Patient Walkthrough -- Risk Evolution

| Hour | Risk Score | Status |
|---|---|---|
| T-12 | 0.23 | Low Risk |
| T-6 | 0.58 | **Elevated -- RescueWindow ALERT** |
| T-0 | 0.94 | Onset |

At T-6, specialist outputs:

| Specialist | Score | Interpretation |
|---|---|---|
| Snapshot (XGBoost) | 0.41 | Current vitals slightly elevated |
| Trajectory (GRU) | 0.72 | Strong rising trend detected |
| Phenotype | Rapid Decline | Archetype: rapid collapse pattern |
| **FUSED OUTPUT** | **0.81** | High confidence alert triggered |

### Phenotype Archetypes Learned (k=8 clusters)

| Archetype | Description |
|---|---|
| Stable | Consistent vitals, low risk |
| Inflammatory Drift | Gradual WBC rise, temperature increase |
| Rapid Collapse | Sharp BP drop, HR spike |
| Oscillating | Unstable, fluctuating vitals |

### Comparison to Literature

| System | AUROC |
|---|---|
| Current EWS (threshold-based) | 0.71-0.81 |
| Prior ML sepsis detection | 0.85-0.97 |
| **RescueWindow (ours)** | **0.87** |

### Clinical Impact (Evidence from Deployed Systems)

| System | Outcome |
|---|---|
| COMPOSER | -1.9% absolute mortality reduction |
| TREWS | -3.3% when providers responded |
| MLASA | +8.3% antibiotics within 1 hour |
| Bundle Compliance | +5% improvement in adherence |

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
- Current sepsis risk (0-1)
- Which model is most concerned
- Patient archetype (e.g., "early_inflammatory", "rapid_collapse")
- **Estimated rescue window** -- how many hours remain before projected onset

---

## Dataset

**PhysioNet 2019 Sepsis Challenge** -- 40,336 patients, hourly time-series, ~40 vitals/labs.

- 2.5M+ hourly windows
- 15M+ data points
- 40+ clinical variables (vitals, labs, demographics)
- Sepsis-3 criteria: SOFA score >=2 with suspected infection

### Download

```bash
python download_data.py
```

Files land in `data/raw/training_setA.csv` and `data/raw/training_setB.csv`.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch.

> **Note**: GPU is required for GRU training at this dataset scale. We used 4x NVIDIA A100 GPUs on SHARCNET. Training on CPU is not feasible for the full dataset.

---

## Train

> **Cannot run without SHARCNET / HPC access** -- see compute note at top of this file.

```bash
python train.py
```

This runs the full pipeline:
1. Load raw PSV files
2. Preprocess + clip physiological values
3. Patient-level 70/15/15 split (no row leakage)
4. Forward-fill + population median imputation (fit on train only)
5. Feature engineering: rolling means/slopes/stds, derived vitals, missingness flags
6. Train Snapshot XGBoost
7. Train Trajectory GRU (PyTorch)
8. Train Phenotype Autoencoder + K-Means
9. Train Fusion XGBoost stacker
10. Evaluate all 4 models vs logistic regression baseline
11. Save comparison table, ROC/PR curves, and all model artifacts

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
| % early by 6h | % of caught patients alerted >=6h before onset |

---

## Project structure

```
rescuewindow/
|-- config/
|-- data/
|-- models/
|   |-- snapshot/
|   |-- trajectory/
|   |-- phenotype/
|   `-- fusion/
|-- evaluation/
|-- inference/
|-- utils/
|-- train.py
|-- predict.py
`-- download_data.py
```

---

## Key design decisions

- **Patient-level split**: No patient appears in both train and test.
- **Imputation discipline**: Population medians computed on train only, applied to val/test.
- **Target**: Sepsis onset within next 6 hours -- not just current label.
- **Stacking safety**: Base models train on train split; fusion stacker trains on val split predictions.
- **Phenotype as soft features**: Distances to all k=8 centroids passed to the stacker.
- **Rescue window**: Defined as prediction_horizon - clinical_action_lag from the first alert.
