"""GRU trajectory specialist — 12-hour window of raw signals."""
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import joblib

from models import BaseModel
from utils.logging import get_logger

log = get_logger(__name__)


class GRUClassifier(nn.Module):
    """GRU encoder → linear classification head."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:    (batch, seq_len, input_dim)
        mask: (batch, seq_len) — 1 for valid timesteps, 0 for padding

        Returns logits (batch,)
        """
        if mask is not None:
            # Compute actual lengths for pack_padded_sequence
            lengths = mask.sum(dim=1).long().cpu()
            lengths = torch.clamp(lengths, min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(x)

        # Take last layer's hidden state
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        out = self.dropout(last_hidden)
        logits = self.head(out).squeeze(-1)  # (batch,)
        return logits


class TrajectoryModel(BaseModel):
    """Wrapper around GRUClassifier with fit/predict_proba interface."""

    def __init__(self, cfg: dict, input_dim: int):
        p = cfg.get("trajectory", {})
        self.input_dim = input_dim
        self.hidden_dim = p.get("hidden_dim", 64)
        self.num_layers = p.get("num_layers", 2)
        self.dropout = p.get("dropout", 0.3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[GRUClassifier] = None
        self._cfg = cfg

    def _build_net(self) -> GRUClassifier:
        return GRUClassifier(
            self.input_dim, self.hidden_dim, self.num_layers, self.dropout
        ).to(self.device)

    def fit(self, X, y, X_val=None, y_val=None, mask=None, mask_val=None, **kwargs):
        """X: (N, seq_len, n_feats), y: (N,)"""
        from models.trajectory.trainer import _training_loop
        self.net = self._build_net()
        self.net = _training_loop(
            self.net, X, y, mask, X_val, y_val, mask_val, self._cfg, self.device
        )
        return self

    @torch.no_grad()
    def predict_proba(self, X, mask=None) -> np.ndarray:
        assert self.net is not None, "Call fit() first"
        self.net.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        m_t = torch.tensor(mask, dtype=torch.float32).to(self.device) if mask is not None else None

        batch_size = 1024
        probs = []
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i : i + batch_size]
            mb = m_t[i : i + batch_size] if m_t is not None else None
            logits = self.net(xb, mb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }, path)
        log.info(f"TrajectoryModel saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TrajectoryModel":
        ckpt = torch.load(path, map_location="cpu")
        obj = cls.__new__(cls)
        obj.input_dim = ckpt["input_dim"]
        obj.hidden_dim = ckpt["hidden_dim"]
        obj.num_layers = ckpt["num_layers"]
        obj.dropout = ckpt["dropout"]
        obj.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj._cfg = {}
        obj.net = GRUClassifier(
            ckpt["input_dim"], ckpt["hidden_dim"], ckpt["num_layers"], ckpt["dropout"]
        ).to(obj.device)
        obj.net.load_state_dict(ckpt["state_dict"])
        obj.net.eval()
        return obj
