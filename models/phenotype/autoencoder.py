"""Autoencoder for learning patient-hour embeddings."""
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn

from utils.logging import get_logger

log = get_logger(__name__)


class Autoencoder(nn.Module):
    """Symmetric encoder-decoder over snapshot feature vectors."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            # Default: taper toward latent_dim
            h = max(latent_dim * 2, 64)
            hidden_dims = [min(input_dim, 256), h]

        # Encoder
        enc_layers = []
        in_d = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_d = h
        enc_layers += [nn.Linear(in_d, latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror)
        dec_layers = []
        in_d = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_d = h
        dec_layers += [nn.Linear(in_d, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(
    X_train: np.ndarray,
    input_dim: int,
    cfg: dict,
    device: torch.device,
) -> Autoencoder:
    p = cfg.get("phenotype", {})
    latent_dim = p.get("latent_dim", 32)
    epochs = p.get("autoencoder_epochs", 50)
    lr = p.get("autoencoder_lr", 0.001)
    batch_size = p.get("autoencoder_batch", 512)

    net = Autoencoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        net.train()
        total_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon, _ = net(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            log.info(f"Autoencoder epoch {epoch+1}/{epochs} | loss={total_loss/len(loader):.5f}")

    return net


@torch.no_grad()
def get_embeddings(net: Autoencoder, X: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    net.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    embeddings = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i : i + batch_size].to(device)
        z = net.encode(xb)
        embeddings.append(z.cpu().numpy())
    return np.concatenate(embeddings)
