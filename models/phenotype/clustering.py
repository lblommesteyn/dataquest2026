"""K-means clustering over latent embeddings + soft distance features."""
import os
from typing import Optional

import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.logging import get_logger

log = get_logger(__name__)

ARCHETYPE_NAMES = {
    0: "stable",
    1: "early_inflammatory",
    2: "rapid_collapse",
    3: "oscillating_unstable",
    4: "hypotensive",
    5: "respiratory_predominant",
    6: "renal_failure_pattern",
    7: "late_onset",
}


class PhenotypeClusterer:
    """K-means on latent embeddings. Exposes soft distance features for fusion."""

    def __init__(self, n_clusters: int = 8, seed: int = 42):
        self.n_clusters = n_clusters
        self.seed = seed
        self.kmeans: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None

    def fit(self, embeddings: np.ndarray) -> "PhenotypeClusterer":
        self.scaler = StandardScaler()
        Z = self.scaler.fit_transform(embeddings)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        self.kmeans.fit(Z)
        log.info(f"K-means fitted: {self.n_clusters} clusters on {len(embeddings):,} embeddings")
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(embeddings)
        return self.kmeans.predict(Z)

    def soft_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Return (N, k) distances to each centroid. Negative = closer."""
        Z = self.scaler.transform(embeddings)
        centroids = self.kmeans.cluster_centers_
        # Euclidean distance from each point to each centroid
        diffs = Z[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (N, k, latent_dim)
        dists = np.linalg.norm(diffs, axis=2)  # (N, k)
        return dists

    def archetype_name(self, cluster_id: int) -> str:
        return ARCHETYPE_NAMES.get(cluster_id, f"cluster_{cluster_id}")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"kmeans": self.kmeans, "scaler": self.scaler, "n_clusters": self.n_clusters}, path)

    @classmethod
    def load(cls, path: str) -> "PhenotypeClusterer":
        state = joblib.load(path)
        obj = cls(n_clusters=state["n_clusters"])
        obj.kmeans = state["kmeans"]
        obj.scaler = state["scaler"]
        return obj
