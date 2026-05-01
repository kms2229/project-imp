"""Hybrid loss functions for QEVC training.

The hybrid loss combines quantum and classical path losses:
    L = λ · L_quantum + (1 - λ) · L_classical

Supports both cross-entropy (VQA-CP) and binary cross-entropy (MIMIC).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridLoss(nn.Module):
    """Weighted combination of quantum and classical classification losses.

    Parameters
    ----------
    task : str
        ``'vqacp'`` for multi-class CE, ``'mimic'`` for multi-label BCE.
    lam : float
        Weight for the quantum path loss. Classical weight = 1 - lam.
    label_smoothing : float
        Label smoothing for cross-entropy (VQA-CP only). Default: 0.0.
    """

    def __init__(
        self,
        task: str = "vqacp",
        lam: float = 0.5,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.task = task
        self.lam = lam
        self.label_smoothing = label_smoothing

        if task in ("vqacp", "mimic"):
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            raise ValueError(f"Unknown task: {task!r}. Expected 'vqacp' or 'mimic'.")

    def forward(
        self,
        q_logits: torch.Tensor,
        c_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute hybrid loss.

        Parameters
        ----------
        q_logits : Tensor (B, n_classes)
            Quantum path logits.
        c_logits : Tensor (B, n_classes)
            Classical path logits.
        targets : Tensor
            Ground-truth labels. (B,) long for VQA-CP, (B, n_codes) float for MIMIC.

        Returns
        -------
        loss : Tensor
            Scalar loss for backpropagation.
        breakdown : dict
            ``{'loss_q': float, 'loss_c': float, 'loss_total': float}``
        """
        loss_q = self.criterion(q_logits, targets)
        loss_c = self.criterion(c_logits, targets)

        loss = self.lam * loss_q + (1 - self.lam) * loss_c

        breakdown = {
            "loss_q": loss_q.item(),
            "loss_c": loss_c.item(),
            "loss_total": loss.item(),
        }
        return loss, breakdown


def get_loss(task: str = "vqacp", lam: float = 0.5, **kwargs) -> HybridLoss:
    """Factory function for creating a configured HybridLoss.

    Parameters
    ----------
    task : str
        ``'vqacp'`` or ``'mimic'``.
    lam : float
        Lambda weight for quantum path.
    **kwargs
        Additional arguments passed to ``HybridLoss``.
    """
    return HybridLoss(task=task, lam=lam, **kwargs)
