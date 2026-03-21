"""Assemble comparison table and write results to disk."""
import os
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import roc_curve, precision_recall_curve

from evaluation.metrics import compute_metrics, find_threshold_at_sensitivity
from evaluation.lead_time import compute_lead_times
from utils.logging import get_logger

log = get_logger(__name__)


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    patient_ids: np.ndarray,
    hours: np.ndarray,
    sensitivity_target: float = 0.80,
) -> Dict:
    metrics = compute_metrics(y_true, y_score, label=name)
    threshold = find_threshold_at_sensitivity(y_true, y_score, sensitivity_target)
    lead = compute_lead_times(patient_ids, hours, y_true, y_score, threshold)
    return {
        "name": name,
        "metrics": metrics,
        "threshold": threshold,
        "lead_time": lead["summary"],
    }


def build_comparison_table(model_results: list) -> pd.DataFrame:
    rows = []
    for r in model_results:
        row = {"model": r["name"]}
        row.update({
            "AUROC": r["metrics"].get(f"{r['name']}_auroc", r["metrics"].get("auroc", float("nan"))),
            "AUPRC": r["metrics"].get(f"{r['name']}_auprc", r["metrics"].get("auprc", float("nan"))),
            "F1@0.5": r["metrics"].get(f"{r['name']}_f1@0.5", float("nan")),
            "Sensitivity@Spec90": r["metrics"].get(f"{r['name']}_sensitivity@spec90", float("nan")),
            "Alert Threshold": r["threshold"],
            "Detection Rate": r["lead_time"].get("detection_rate", float("nan")),
            "Mean Lead Time (h)": r["lead_time"].get("mean_lead_time_h", float("nan")),
            "Median Lead Time (h)": r["lead_time"].get("median_lead_time_h", float("nan")),
            "% Early by 6h": r["lead_time"].get("pct_early_by_6h", float("nan")),
        })
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def save_results(
    model_results: list,
    all_scores: Dict[str, np.ndarray],
    y_true: np.ndarray,
    results_dir: str,
):
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Comparison table
    table = build_comparison_table(model_results)
    table_path = os.path.join(results_dir, "comparison_table.csv")
    table.to_csv(table_path)
    log.info(f"\n{table.to_string()}")
    log.info(f"Comparison table saved to {table_path}")

    # ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["steelblue", "darkorange", "green", "red"]
    for i, (name, scores) in enumerate(all_scores.items()):
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = roc_auc_score(y_true, scores)
        ap = average_precision_score(y_true, scores)
        c = colors[i % len(colors)]
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=c)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=c)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title("ROC Curves")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend(fontsize=9)

    axes[1].set_title("Precision-Recall Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "curves.png"), dpi=150)
    plt.close(fig)

    # Save raw results JSON
    clean_results = []
    for r in model_results:
        clean_results.append({
            "name": r["name"],
            "threshold": r["threshold"],
            "lead_time": r["lead_time"],
            "metrics": {k: (v if not (isinstance(v, float) and np.isnan(v)) else None)
                        for k, v in r["metrics"].items()},
        })
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(clean_results, f, indent=2)

    log.info(f"All results saved to {results_dir}")
# iteration 7: reporter updated
# iteration 17: reporter updated
# iteration 27: reporter updated
