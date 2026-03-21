"""Detect the rescue window from a patient's score time-series."""
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from inference.scorer import PatientScores
from models.phenotype.clustering import ARCHETYPE_NAMES
from utils.logging import get_logger

log = get_logger(__name__)

# Clinical assumption: once alerted, intervention takes ~2 hours to implement
CLINICAL_ACTION_LAG_HOURS = 2


@dataclass
class RescueWindowResult:
    patient_id: str
    alert_issued: bool
    alert_hour: Optional[int]
    predicted_onset_hour: Optional[int]
    window_duration_hours: Optional[float]
    confidence: Optional[float]
    phenotype_cluster: Optional[int]
    phenotype_name: Optional[str]
    risk_trend: str  # "rising", "falling", "stable", "peak"
    all_risks: List[float] = field(default_factory=list)

    def summary(self) -> str:
        if not self.alert_issued:
            return f"[{self.patient_id}] No alert issued. Max risk: {max(self.all_risks, default=0):.3f}"
        return (
            f"[{self.patient_id}] ALERT at hour {self.alert_hour} | "
            f"Predicted onset: hour {self.predicted_onset_hour} | "
            f"Rescue window: {self.window_duration_hours:.1f}h | "
            f"Confidence: {self.confidence:.2f} | "
            f"Phenotype: {self.phenotype_name} | "
            f"Trend: {self.risk_trend}"
        )


def _classify_trend(risks: np.ndarray, window: int = 4) -> str:
    if len(risks) < 2:
        return "stable"
    recent = risks[-window:]
    if len(recent) < 2:
        return "stable"
    slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
    if slope > 0.02:
        return "rising"
    elif slope < -0.02:
        return "falling"
    elif risks[-1] >= 0.7:
        return "peak"
    return "stable"


def detect_rescue_window(
    scores: PatientScores,
    alert_threshold: float = 0.4,
    prediction_horizon_hours: int = 6,
) -> RescueWindowResult:
    """
    Determine whether and when the model issues a rescue alert,
    and estimate the rescue window duration.
    """
    risks = scores.risk_series
    hours = scores.hours
    hour_scores = scores.hour_scores

    if len(risks) == 0:
        return RescueWindowResult(
            patient_id=scores.patient_id,
            alert_issued=False, alert_hour=None,
            predicted_onset_hour=None, window_duration_hours=None,
            confidence=None, phenotype_cluster=None, phenotype_name=None,
            risk_trend="stable", all_risks=[],
        )

    # Find first alert hour
    alert_mask = risks >= alert_threshold
    alert_hours_idx = np.where(alert_mask)[0]

    trend = _classify_trend(risks)

    if len(alert_hours_idx) == 0:
        return RescueWindowResult(
            patient_id=scores.patient_id,
            alert_issued=False, alert_hour=None,
            predicted_onset_hour=None, window_duration_hours=None,
            confidence=float(risks.max()), phenotype_cluster=None, phenotype_name=None,
            risk_trend=trend, all_risks=risks.tolist(),
        )

    first_alert_idx = alert_hours_idx[0]
    alert_hour = int(hours[first_alert_idx])
    confidence = float(risks[first_alert_idx])

    # Predicted onset = alert_hour + expected_horizon
    # In reality, the model is trained to predict onset within 6h,
    # so best estimate is alert_hour + horizon/2 (center of window)
    predicted_onset_hour = alert_hour + prediction_horizon_hours // 2

    # Rescue window = time from alert to predicted onset minus clinical action lag
    window_duration = prediction_horizon_hours - CLINICAL_ACTION_LAG_HOURS

    # Phenotype at alert hour
    hs = hour_scores[first_alert_idx]
    cluster = hs.phenotype_cluster
    pheno_name = hs.phenotype_name

    result = RescueWindowResult(
        patient_id=scores.patient_id,
        alert_issued=True,
        alert_hour=alert_hour,
        predicted_onset_hour=predicted_onset_hour,
        window_duration_hours=float(window_duration),
        confidence=confidence,
        phenotype_cluster=cluster,
        phenotype_name=pheno_name,
        risk_trend=trend,
        all_risks=risks.tolist(),
    )

    log.info(result.summary())
    return result
# iteration 9: rescue window updated
# iteration 19: rescue window updated
# iteration 29: rescue window updated
# iteration 39: rescue window updated
