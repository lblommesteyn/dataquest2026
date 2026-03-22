"""PyTorch training loop for GRU trajectory model."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from models.trajectory.model import GRUClassifier
from evaluation.metrics import compute_metrics
from utils.logging import get_logger

log = get_logger(__name__)


def _training_loop(
    net: GRUClassifier,
    X_tr, y_tr, mask_tr,
    X_val, y_val, mask_val,
    cfg: dict,
    device: torch.device,
) -> GRUClassifier:
    p = cfg.get("trajectory", {})
    epochs = p.get("epochs", 30)
    batch_size = p.get("batch_size", 256)
    lr = p.get("lr", 0.001)
    grad_clip = p.get("grad_clip", 1.0)

    # Tensors
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    mt = torch.tensor(mask_tr, dtype=torch.float32) if mask_tr is not None else torch.ones(len(Xt), Xt.shape[1])

    # Weighted sampler for class imbalance
    n_pos = int(yt.sum())
    n_neg = len(yt) - n_pos
    weights = torch.where(yt > 0, torch.tensor(1.0 / n_pos), torch.tensor(1.0 / n_neg))
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    dataset = TensorDataset(Xt, mt, yt)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    criterion = nn.BCEWithLogitsLoss()

    # Validation tensors
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    mv = torch.tensor(mask_val, dtype=torch.float32).to(device) if mask_val is not None else None

    best_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        net.train()
        total_loss = 0.0
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = net(xb, mb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        net.eval()
        with torch.no_grad():
            batch_size_val = 2048
            probs_val = []
            for i in range(0, len(Xv), batch_size_val):
                xb = Xv[i : i + batch_size_val]
                mb = mv[i : i + batch_size_val] if mv is not None else None
                logits = net(xb, mb)
                probs_val.append(torch.sigmoid(logits).cpu().numpy())
            probs_val = np.concatenate(probs_val)

        metrics = compute_metrics(y_val, probs_val, label="")
        val_auc = metrics.get("auroc", 0.0)
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        log.info(
            f"Epoch {epoch+1}/{epochs} | loss={total_loss/len(loader):.4f} | val_auroc={val_auc:.4f}"
        )

    if best_state:
        net.load_state_dict(best_state)
        log.info(f"Loaded best weights (val_auroc={best_auc:.4f})")

    return net


def train_trajectory(train_data: dict, val_data: dict, cfg: dict):
    from models.trajectory.model import TrajectoryModel

    input_dim = train_data["X_traj"].shape[2]
    model = TrajectoryModel(cfg, input_dim)
    model.fit(
        train_data["X_traj"], train_data["y"],
        X_val=val_data["X_traj"], y_val=val_data["y"],
        mask=train_data["traj_mask"], mask_val=val_data["traj_mask"],
    )
    return model
