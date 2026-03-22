"""AUROC, AUPRC, F1, and operating-point metrics."""
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label: str = "",
    thresholds: tuple = (0.3, 0.4, 0.5),
) -> Dict[str, float]:
    """Return flat dict of classification metrics."""
    results = {}
    prefix = f"{label}_" if label else ""

    if len(np.unique(y_true)) < 2:
        return {f"{prefix}auroc": float("nan"), f"{prefix}auprc": float("nan")}

    results[f"{prefix}auroc"] = roc_auc_score(y_true, y_score)
    results[f"{prefix}auprc"] = average_precision_score(y_true, y_score)

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        results[f"{prefix}f1@{t}"] = f1_score(y_true, y_pred, zero_division=0)

    # Recall at 90% precision
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    high_prec_idx = np.where(prec >= 0.90)[0]
    if len(high_prec_idx) > 0:
        results[f"{prefix}recall@prec90"] = float(rec[high_prec_idx[-1]])
    else:
        results[f"{prefix}recall@prec90"] = 0.0

    # Sensitivity at 90% specificity
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1 - fpr
    high_spec_idx = np.where(spec >= 0.90)[0]
    if len(high_spec_idx) > 0:
        results[f"{prefix}sensitivity@spec90"] = float(tpr[high_spec_idx[-1]])
    else:
        results[f"{prefix}sensitivity@spec90"] = 0.0

    return results


def find_threshold_at_sensitivity(
    y_true: np.ndarray, y_score: np.ndarray, target_sensitivity: float = 0.80
) -> float:
    """Return the score threshold that achieves approximately target_sensitivity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Find lowest threshold where TPR >= target
    eligible = np.where(tpr >= target_sensitivity)[0]
    if len(eligible) == 0:
        return 0.0
    idx = eligible[0]  # first threshold achieving target sensitivity
    return float(thresholds[idx])
