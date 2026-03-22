"""Train autoencoder + fit k-means, return phenotype model."""
import os
import torch
import numpy as np
import joblib

from models.phenotype.autoencoder import Autoencoder, train_autoencoder, get_embeddings
from models.phenotype.clustering import PhenotypeClusterer
from utils.logging import get_logger

log = get_logger(__name__)


class PhenotypeModel:
    """Combined autoencoder + clusterer with a unified predict interface."""

    def __init__(self):
        self.autoencoder: Autoencoder = None
        self.clusterer: PhenotypeClusterer = None
        self._device = None

    def fit(self, X_train: np.ndarray, cfg: dict) -> "PhenotypeModel":
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X_train.shape[1]

        # 1) Train autoencoder
        self.autoencoder = train_autoencoder(X_train, input_dim, cfg, self._device)

        # 2) Embed training data
        embeddings = get_embeddings(self.autoencoder, X_train, self._device)

        # 3) Cluster
        n_clusters = cfg.get("phenotype", {}).get("n_clusters", 8)
        self.clusterer = PhenotypeClusterer(n_clusters=n_clusters, seed=cfg.get("seed", 42))
        self.clusterer.fit(embeddings)

        log.info("PhenotypeModel fitted")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return soft cluster distances (N, k) as fusion features."""
        embeddings = get_embeddings(self.autoencoder, X, self._device)
        return self.clusterer.soft_distances(embeddings)

    def predict_cluster(self, X: np.ndarray) -> np.ndarray:
        """Return hard cluster assignments (N,)."""
        embeddings = get_embeddings(self.autoencoder, X, self._device)
        return self.clusterer.predict(embeddings)

    def save(self, path_prefix: str):
        os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
        torch.save(self.autoencoder.state_dict(), path_prefix + "_ae.pt")
        # Save input_dim and architecture for reconstruction
        joblib.dump({
            "input_dim": self.autoencoder.encoder[0].in_features,
            "latent_dim": self.autoencoder.encoder[-1].out_features,
        }, path_prefix + "_ae_meta.pkl")
        self.clusterer.save(path_prefix + "_clusterer.pkl")
        log.info(f"PhenotypeModel saved to {path_prefix}*")

    @classmethod
    def load(cls, path_prefix: str) -> "PhenotypeModel":
        meta = joblib.load(path_prefix + "_ae_meta.pkl")
        ae = Autoencoder(meta["input_dim"], meta["latent_dim"])
        ae.load_state_dict(torch.load(path_prefix + "_ae.pt", map_location="cpu"))
        ae.eval()

        clusterer = PhenotypeClusterer.load(path_prefix + "_clusterer.pkl")

        obj = cls()
        obj.autoencoder = ae
        obj.clusterer = clusterer
        obj._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj.autoencoder = obj.autoencoder.to(obj._device)
        return obj


def train_phenotype(train_data: dict, cfg: dict) -> PhenotypeModel:
    model = PhenotypeModel()
    model.fit(train_data["X_snapshot"], cfg)
    return model
