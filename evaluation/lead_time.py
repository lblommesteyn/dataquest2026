"""Compute early-warning lead time before sepsis onset."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logging import get_logger

log = get_logger(__name__)


def compute_lead_times(
    patient_ids: np.ndarray,
    hours: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict:
    """
    For each patient who develops sepsis, find the hour when model first alerts
    (score >= threshold) and compute how many hours before sepsis onset that is.

    Returns dict with per-patient lead times and summary stats.
    """
    results = []
    unique_pids = np.unique(patient_ids)

    for pid in unique_pids:
        mask = patient_ids == pid
        h = hours[mask]
        y = y_true[mask]
        s = y_score[mask]

        # Does this patient develop sepsis?
        if y.max() == 0:
            continue

        # Find sepsis onset hour (first hour with label=1)
        onset_idx = int(np.argmax(y > 0))
        onset_hour = int(h[onset_idx])

        # Find first alert hour (score >= threshold)
        alert_hours = h[s >= threshold]
        if len(alert_hours) == 0:
            # Missed — no alert issued
            results.append({
                "patient_id": pid,
                "onset_hour": onset_hour,
                "alert_hour": None,
                "lead_time": None,
                "missed": True,
            })
            continue

        alert_hour = int(alert_hours[0])
        lead_time = onset_hour - alert_hour  # positive = alert before onset

        results.append({
            "patient_id": pid,
            "onset_hour": onset_hour,
            "alert_hour": alert_hour,
            "lead_time": lead_time,
            "missed": False,
        })

    df = pd.DataFrame(results)
    caught = df[~df["missed"]]
    lead_times = caught["lead_time"].dropna().values

    summary = {
        "n_sepsis_patients": len(df),
        "n_caught": int((~df["missed"]).sum()),
        "n_missed": int(df["missed"].sum()),
        "detection_rate": float((~df["missed"]).mean()) if len(df) > 0 else 0.0,
        "mean_lead_time_h": float(lead_times.mean()) if len(lead_times) > 0 else 0.0,
        "median_lead_time_h": float(np.median(lead_times)) if len(lead_times) > 0 else 0.0,
        "pct_early_by_3h": float((lead_times >= 3).mean()) if len(lead_times) > 0 else 0.0,
        "pct_early_by_6h": float((lead_times >= 6).mean()) if len(lead_times) > 0 else 0.0,
        "pct_late_alerts": float((lead_times < 0).mean()) if len(lead_times) > 0 else 0.0,
    }

    log.info(
        f"Lead time analysis: {summary['n_caught']}/{summary['n_sepsis_patients']} caught, "
        f"mean lead={summary['mean_lead_time_h']:.1f}h"
    )

    return {"summary": summary, "per_patient": df}
