"""Per-patient-hour scoring using all four models."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class HourScore:
    hour: int
    snapshot_risk: float
    trajectory_risk: float
    fusion_risk: float
    phenotype_cluster: int
    phenotype_name: str
    alert: bool


@dataclass
class PatientScores:
    patient_id: str
    hour_scores: List[HourScore] = field(default_factory=list)

    @property
    def risk_series(self) -> np.ndarray:
        return np.array([h.fusion_risk for h in self.hour_scores])

    @property
    def hours(self) -> np.ndarray:
        return np.array([h.hour for h in self.hour_scores])


class RescueWindowScorer:
    """Load all trained models and score a patient's trajectory in real-time."""

    def __init__(self, models_dir: str, alert_threshold: float = 0.4):
        self.models_dir = Path(models_dir)
        self.alert_threshold = alert_threshold
        self._loaded = False
        self.snap_model = None
        self.traj_model = None
        self.pheno_model = None
        self.fusion_model = None
        self.imputer = None

    def load_models(self):
        from models.snapshot.model import SnapshotModel
        from models.trajectory.model import TrajectoryModel
        from models.phenotype.trainer import PhenotypeModel
        from models.fusion.model import FusionModel
        from data.imputer import PatientImputer

        log.info(f"Loading models from {self.models_dir}")
        self.snap_model = SnapshotModel.load(str(self.models_dir / "snapshot.pkl"))
        self.traj_model = TrajectoryModel.load(str(self.models_dir / "trajectory.pt"))
        self.pheno_model = PhenotypeModel.load(str(self.models_dir / "phenotype"))
        self.fusion_model = FusionModel.load(str(self.models_dir / "fusion.pkl"))
        self.imputer = PatientImputer.load(str(self.models_dir / "imputer.pkl"))
        self._loaded = True
        log.info("All models loaded")

    def score_patient(
        self,
        patient_df: pd.DataFrame,
        feature_cols: List[str],
        traj_cols: List[str],
        seq_len: int = 12,
    ) -> PatientScores:
        """Score a single patient's DataFrame row-by-row."""
        assert self._loaded, "Call load_models() first"

        from models.fusion.model import build_meta_features
        from data.windower import _extract_trajectory
        from models.phenotype.clustering import ARCHETYPE_NAMES

        # Ensure features exist
        for col in feature_cols:
            if col not in patient_df.columns:
                patient_df[col] = 0.0

        feat_cols_avail = [c for c in feature_cols if c in patient_df.columns]
        X_snap = patient_df[feat_cols_avail].values.astype(np.float32)
        X_traj, traj_mask = _extract_trajectory(patient_df, traj_cols, seq_len)

        snap_scores = self.snap_model.predict_proba(X_snap)
        traj_scores = self.traj_model.predict_proba(X_traj, mask=traj_mask)
        pheno_dists = self.pheno_model.predict_proba(X_snap)
        pheno_clusters = self.pheno_model.predict_cluster(X_snap)

        X_meta = build_meta_features(snap_scores, traj_scores, pheno_dists)
        fusion_scores = self.fusion_model.predict_proba(X_meta)

        pid = "patient"
        result = PatientScores(patient_id=pid)
        for i in range(len(fusion_scores)):
            cluster = int(pheno_clusters[i])
            result.hour_scores.append(HourScore(
                hour=i,
                snapshot_risk=float(snap_scores[i]),
                trajectory_risk=float(traj_scores[i]),
                fusion_risk=float(fusion_scores[i]),
                phenotype_cluster=cluster,
                phenotype_name=ARCHETYPE_NAMES.get(cluster, f"cluster_{cluster}"),
                alert=float(fusion_scores[i]) >= self.alert_threshold,
            ))

        return result
# iteration 8: inference updated
# iteration 18: inference updated
