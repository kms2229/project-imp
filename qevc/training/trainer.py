"""QEVC Trainer: training loop with early stopping, checkpointing, and logging.

Handles the full training lifecycle including:
  - Per-epoch training and validation
  - Early stopping based on validation loss
  - Best-model checkpointing
  - Per-epoch QFS computation
  - Training curve saving
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from qevc.quantum.qfs import quantum_feature_score
from qevc.training.losses import HybridLoss


class QEVCTrainer:
    """Full training pipeline for the QEVC hybrid model.

    Parameters
    ----------
    model : nn.Module
        The ``QEVCModel`` instance.
    config : object
        ``QEVCConfig`` with training hyperparameters.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    device : torch.device
        Target device.
    task : str
        ``'vqacp'`` or ``'mimic'``.
    checkpoint_dir : str or Path
        Directory for saving checkpoints.
    results_dir : str or Path
        Directory for saving training curves.
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        task: str = "vqacp",
        checkpoint_dir: str | Path = "qevc/checkpoints",
        results_dir: str | Path = "qevc/results",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.task = task
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = HybridLoss(task=task, lam=config.lam)

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # Learning rate scheduler (reduce on plateau)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Training history
        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "qfs": [],
            "lr": [],
        }

    # ------------------------------------------------------------------ #
    # Main training loop
    # ------------------------------------------------------------------ #

    def train(self) -> dict:
        """Run the full training loop.

        Returns
        -------
        history : dict
            Training history with per-epoch metrics.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"\n{'='*60}")
        print(f"Training QEVC — {self.task.upper()}")
        print(f"Device: {self.device} | Epochs: {self.config.epochs} | "
              f"Patience: {self.config.patience}")
        print(f"Lambda: {self.config.lam} | LR: {self.config.lr} | "
              f"Batch: {self.config.batch_size}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()

            # --- Train ---
            train_metrics = self._train_epoch()

            # --- Validate ---
            val_metrics = self._validate_epoch()

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            # Log
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["val_acc"].append(val_metrics["acc"])
            self.history["qfs"].append(val_metrics.get("qfs", 0.0))
            self.history["lr"].append(lr)

            print(
                f"Epoch {epoch:3d}/{self.config.epochs} │ "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} │ "
                f"train_acc={train_metrics['acc']:.3f} "
                f"val_acc={val_metrics['acc']:.3f} │ "
                f"QFS={val_metrics.get('qfs', 0):.3f} │ "
                f"lr={lr:.2e} │ {elapsed:.1f}s"
            )

            # --- LR schedule ---
            self.scheduler.step(val_metrics["loss"])

            # --- Early stopping + checkpointing ---
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(patience={self.config.patience})")
                    break

        # Save training curves
        self._save_history()
        print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
        return self.history

    # ------------------------------------------------------------------ #
    # Single epoch routines
    # ------------------------------------------------------------------ #

    def _train_epoch(self) -> dict:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.train_loader, desc="  Train", leave=False):
            features = batch["features"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            q_logits, c_logits, _ = self.model(features)
            loss, _ = self.criterion(q_logits, c_logits, labels)

            # Gradient clipping for quantum parameter stability
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * features.size(0)

            # Accuracy (use quantum logits)
            if self.task == "vqacp":
                preds = q_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
            else:
                preds = (q_logits > 0).float()
                correct += ((preds == labels).all(dim=1)).sum().item()
            total += features.size(0)

        return {
            "loss": total_loss / max(total, 1),
            "acc": correct / max(total, 1),
        }

    @torch.no_grad()
    def _validate_epoch(self) -> dict:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_q_features = []
        all_labels = []
        all_groups = []

        for batch in tqdm(self.val_loader, desc="  Val  ", leave=False):
            features = batch["features"].to(self.device)
            labels = batch["label"].to(self.device)
            groups = batch["group"]

            q_logits, c_logits, q_features = self.model(features)
            loss, _ = self.criterion(q_logits, c_logits, labels)

            total_loss += loss.item() * features.size(0)

            if self.task == "vqacp":
                preds = q_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
            else:
                preds = (q_logits > 0).float()
                correct += ((preds == labels).all(dim=1)).sum().item()
            total += features.size(0)

            all_q_features.append(q_features.cpu())
            all_labels.append(labels.cpu())
            all_groups.append(groups)

        # Compute QFS on validation set
        q_feat_cat = torch.cat(all_q_features)
        labels_cat = torch.cat(all_labels)
        groups_cat = torch.cat(all_groups)
        qfs = quantum_feature_score(q_feat_cat, labels_cat, groups_cat)

        return {
            "loss": total_loss / max(total, 1),
            "acc": correct / max(total, 1),
            "qfs": qfs,
        }

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def _save_checkpoint(
        self, epoch: int, metrics: dict, is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        tag = "best" if is_best else f"epoch_{epoch}"
        path = self.checkpoint_dir / f"qevc_{self.task}_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": vars(self.config),
            },
            path,
        )

    def _save_history(self) -> None:
        """Save training history to JSON."""
        path = self.results_dir / f"training_curves_{self.task}.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training curves saved to {path}")

    # ------------------------------------------------------------------ #
    # Evaluation on test set
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict:
        """Run full evaluation on a test set.

        Returns
        -------
        results : dict
            Full metrics including EOD, IBD-F1, AUROC, QFS.
        """
        from qevc.evaluation.metrics import compute_all_metrics

        self.model.eval()

        all_q_logits = []
        all_labels = []
        all_groups = []
        all_q_features = []

        for batch in tqdm(test_loader, desc="Evaluating"):
            features = batch["features"].to(self.device)
            labels = batch["label"]
            groups = batch["group"]

            q_logits, _, q_features = self.model(features)

            all_q_logits.append(q_logits.cpu())
            all_labels.append(labels)
            all_groups.append(groups)
            all_q_features.append(q_features.cpu())

        preds = torch.cat(all_q_logits)
        labels = torch.cat(all_labels)
        groups = torch.cat(all_groups)
        q_features = torch.cat(all_q_features)

        results = compute_all_metrics(
            preds=preds,
            labels=labels,
            groups=groups,
            quantum_outputs=q_features,
            task=self.task,
        )

        print("\n--- Evaluation Results ---")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

        return results
