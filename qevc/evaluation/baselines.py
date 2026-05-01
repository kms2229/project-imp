"""Classical baselines for comparison with QEVC.

Baselines:
  1. **SVM-RBF** — Support Vector Machine with RBF kernel (sklearn)
  2. **MLP** — 3-layer Multi-Layer Perceptron (PyTorch)
  3. **AdvDeb** — Adversarial Debiasing (classifier + group adversary)
  4. **DBA** — Distribution-Balanced Approach (group-reweighted loss)

All baselines operate on the same PCA-fused features as QEVC for fair comparison.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm


# ======================================================================== #
# 1. SVM-RBF Baseline
# ======================================================================== #

class SVMBaseline:
    """Support Vector Machine with RBF kernel.

    Uses sklearn's SVC. For multi-label (MIMIC), wraps in OneVsRestClassifier.
    """

    def __init__(self, task: str = "vqacp", C: float = 1.0, gamma: str = "scale"):
        self.task = task
        if task == "mimic":
            self.model = OneVsRestClassifier(SVC(C=C, gamma=gamma, kernel="rbf"))
        else:
            self.model = SVC(C=C, gamma=gamma, kernel="rbf")

    def train(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
        """Fit the SVM on training data."""
        print("Training SVM-RBF baseline...")
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        return self.model.predict(X)


# ======================================================================== #
# 2. MLP Baseline
# ======================================================================== #

class _MLPNet(nn.Module):
    """Simple 3-layer MLP for classification."""

    def __init__(self, n_input: int, n_classes: int, task: str = "vqacp"):
        super().__init__()
        self.task = task
        self.net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class MLPBaseline:
    """3-layer MLP classifier baseline.

    Parameters
    ----------
    n_input : int
        Input feature dimension (n_pca).
    n_classes : int
        Number of output classes / codes.
    task : str
        ``'vqacp'`` or ``'mimic'``.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    """

    def __init__(
        self,
        n_input: int = 32,
        n_classes: int = 3129,
        task: str = "vqacp",
        lr: float = 0.001,
        epochs: int = 30,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
    ):
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size

        if device is None:
            from qevc.configs.config import get_device
            device = get_device()
        self.device = device

        self.model = _MLPNet(n_input, n_classes, task).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if task in ("vqacp", "mimic"):
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown task: {task}")

    def train(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
        """Fit MLP on training data."""
        print("Training MLP baseline...")
        X_t = torch.from_numpy(X).float()
        if self.task in ("vqacp", "mimic"):
            y_t = torch.from_numpy(y).long()
        else:
            raise ValueError(f"Unknown task: {self.task}")

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{self.epochs} — loss: {total_loss/len(loader):.4f}")

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        self.model.eval()
        X_t = torch.from_numpy(X).float().to(self.device)
        logits = self.model(X_t)
        if self.task in ("vqacp", "mimic"):
            return logits.argmax(dim=1).cpu().numpy()
        else:
            raise ValueError(f"Unknown task: {self.task}")

    @torch.no_grad()
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Get raw logits for AUROC computation."""
        self.model.eval()
        X_t = torch.from_numpy(X).float().to(self.device)
        return self.model(X_t).cpu().numpy()


# ======================================================================== #
# 3. Adversarial Debiasing (AdvDeb)
# ======================================================================== #

class _AdvDebNet(nn.Module):
    """Classifier + adversary for adversarial debiasing."""

    def __init__(self, n_input: int, n_classes: int, n_groups: int):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # Task classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )
        # Group adversary head (tries to predict protected group)
        self.adversary = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_groups),
        )

    def forward(self, x):
        shared = self.shared(x)
        task_logits = self.classifier(shared)
        group_logits = self.adversary(shared)
        return task_logits, group_logits


class AdvDebBaseline:
    """Adversarial Debiasing baseline.

    Trains a classifier while adversarially minimizing group prediction
    accuracy. The adversary gradient is flipped (gradient reversal)
    so the shared features become group-invariant.

    Parameters
    ----------
    n_input : int
        Input feature dimension.
    n_classes : int
        Number of task classes.
    n_groups : int
        Number of protected groups.
    task : str
        ``'vqacp'`` or ``'mimic'``.
    adv_weight : float
        Weight of the adversarial loss. Default: 1.0.
    """

    def __init__(
        self,
        n_input: int = 32,
        n_classes: int = 3129,
        n_groups: int = 3,
        task: str = "vqacp",
        adv_weight: float = 1.0,
        lr: float = 0.001,
        epochs: int = 30,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
    ):
        self.task = task
        self.adv_weight = adv_weight
        self.epochs = epochs
        self.batch_size = batch_size

        if device is None:
            from qevc.configs.config import get_device
            device = get_device()
        self.device = device

        self.model = _AdvDebNet(n_input, n_classes, n_groups).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if task in ("vqacp", "mimic"):
            self.task_criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown task: {task}")
        self.group_criterion = nn.CrossEntropyLoss()

    def train(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """Fit with adversarial debiasing."""
        print("Training AdvDeb baseline...")
        X_t = torch.from_numpy(X).float()
        if self.task in ("vqacp", "mimic"):
            y_t = torch.from_numpy(y).long()
        else:
            raise ValueError(f"Unknown task: {self.task}")
        g_t = torch.from_numpy(groups).long()

        dataset = TensorDataset(X_t, y_t, g_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            for xb, yb, gb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                gb = gb.to(self.device)

                self.optimizer.zero_grad()
                task_logits, group_logits = self.model(xb)

                # Task loss: minimize
                task_loss = self.task_criterion(task_logits, yb)
                # Adversarial loss: maximize (so we subtract it)
                group_loss = self.group_criterion(group_logits, gb)

                loss = task_loss - self.adv_weight * group_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{self.epochs} — loss: {total_loss/len(loader):.4f}")

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.from_numpy(X).float().to(self.device)
        task_logits, _ = self.model(X_t)
        if self.task in ("vqacp", "mimic"):
            return task_logits.argmax(dim=1).cpu().numpy()
        else:
            raise ValueError(f"Unknown task: {self.task}")

    @torch.no_grad()
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.from_numpy(X).float().to(self.device)
        task_logits, _ = self.model(X_t)
        return task_logits.cpu().numpy()


# ======================================================================== #
# 4. Distribution-Balanced Approach (DBA)
# ======================================================================== #

class DBABaseline:
    """Distribution-Balanced Approach baseline.

    Re-weights the loss per sample based on bias group frequency,
    so under-represented groups receive higher loss weight.

    Parameters
    ----------
    See MLPBaseline for parameter descriptions.
    """

    def __init__(
        self,
        n_input: int = 32,
        n_classes: int = 3129,
        task: str = "vqacp",
        lr: float = 0.001,
        epochs: int = 30,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
    ):
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size

        if device is None:
            from qevc.configs.config import get_device
            device = get_device()
        self.device = device

        self.model = _MLPNet(n_input, n_classes, task).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """Fit with distribution-balanced reweighting."""
        print("Training DBA baseline...")

        # Compute per-group weights (inverse frequency)
        unique_groups, group_counts = np.unique(groups, return_counts=True)
        total = len(groups)
        weight_map = {g: total / (len(unique_groups) * c) for g, c in zip(unique_groups, group_counts)}
        sample_weights = np.array([weight_map[g] for g in groups], dtype=np.float32)

        X_t = torch.from_numpy(X).float()
        if self.task in ("vqacp", "mimic"):
            y_t = torch.from_numpy(y).long()
        else:
            raise ValueError(f"Unknown task: {self.task}")
        w_t = torch.from_numpy(sample_weights).float()

        dataset = TensorDataset(X_t, y_t, w_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            for xb, yb, wb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                wb = wb.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(xb)

                if self.task in ("vqacp", "mimic"):
                    loss = nn.functional.cross_entropy(logits, yb, reduction="none")
                else:
                    raise ValueError(f"Unknown task: {self.task}")

                # Apply distribution-balanced weights
                loss = (loss * wb).mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{self.epochs} — loss: {total_loss/len(loader):.4f}")

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.from_numpy(X).float().to(self.device)
        logits = self.model(X_t)
        if self.task in ("vqacp", "mimic"):
            return logits.argmax(dim=1).cpu().numpy()
        else:
            raise ValueError(f"Unknown task: {self.task}")

    @torch.no_grad()
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.from_numpy(X).float().to(self.device)
        return self.model(X_t).cpu().numpy()
