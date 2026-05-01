"""Evaluation metrics for QEVC: EOD, IBD-F1, AUROC, and QFS.

All metrics support both multi-class (VQA-CP) and multi-label (MIMIC) tasks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Equalized Odds Difference (EOD)
# ---------------------------------------------------------------------------

def equalized_odds_difference(
    preds: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Compute Equalized Odds Difference (EOD).

    EOD measures the maximum disparity in True Positive Rate (TPR)
    and False Positive Rate (FPR) across protected groups.

    Lower EOD is better (0 = perfectly equalized).

    For multi-class: converts to per-class binary and averages.

    Parameters
    ----------
    preds : ndarray (N,) or (N, C)
        Predicted class indices or multi-hot predictions.
    labels : ndarray (N,) or (N, C)
        True labels.
    groups : ndarray (N,)
        Group assignments.

    Returns
    -------
    eod : float
        Equalized Odds Difference in [0, 1+].
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return 0.0

    # Convert to binary per-class if needed
    if preds.ndim == 1:
        # Multi-class → binary for each unique label
        unique_labels = np.unique(labels)
        eods = []
        for lab in unique_labels[:50]:  # cap to avoid expensive computation
            p_bin = (preds == lab).astype(float)
            l_bin = (labels == lab).astype(float)
            eods.append(_binary_eod(p_bin, l_bin, groups, unique_groups))
        return float(np.mean(eods)) if eods else 0.0
    else:
        # Multi-label: compute per-column and average
        eods = []
        for c in range(preds.shape[1]):
            eods.append(_binary_eod(preds[:, c], labels[:, c], groups, unique_groups))
        return float(np.mean(eods))


def _binary_eod(
    preds: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    unique_groups: np.ndarray,
) -> float:
    """Compute EOD for a single binary classification task."""
    tprs = []
    fprs = []

    for g in unique_groups:
        mask = groups == g
        g_preds = preds[mask]
        g_labels = labels[mask]

        # True Positive Rate
        pos_mask = g_labels == 1
        if pos_mask.sum() > 0:
            tpr = g_preds[pos_mask].mean()
        else:
            tpr = 0.0
        tprs.append(tpr)

        # False Positive Rate
        neg_mask = g_labels == 0
        if neg_mask.sum() > 0:
            fpr = g_preds[neg_mask].mean()
        else:
            fpr = 0.0
        fprs.append(fpr)

    tpr_diff = max(tprs) - min(tprs)
    fpr_diff = max(fprs) - min(fprs)

    return (tpr_diff + fpr_diff) / 2.0


# ---------------------------------------------------------------------------
# Intra-Bias Discrepancy F1 (IBD-F1)
# ---------------------------------------------------------------------------

def ibd_f1(
    preds: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Compute Intra-Bias Discrepancy F1 (IBD-F1).

    Measures the disparity in F1 scores across bias groups.
    Lower IBD-F1 indicates less performance disparity (fairer model).

    IBD-F1 = max_group(F1) - min_group(F1)

    Parameters
    ----------
    preds, labels, groups : ndarray
        Same format as `equalized_odds_difference`.

    Returns
    -------
    ibd : float
        F1 disparity in [0, 1]. Lower is better.
    """
    from sklearn.metrics import f1_score

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return 0.0

    f1_scores = []
    for g in unique_groups:
        mask = groups == g
        g_preds = preds[mask]
        g_labels = labels[mask]

        if len(np.unique(g_labels)) < 2 and g_preds.ndim == 1:
            # Can't compute F1 with only one class
            continue

        if g_preds.ndim == 1:
            f1 = f1_score(g_labels, g_preds, average="macro", zero_division=0)
        else:
            f1 = f1_score(g_labels, g_preds, average="micro", zero_division=0)

        f1_scores.append(f1)

    if len(f1_scores) < 2:
        return 0.0

    return float(max(f1_scores) - min(f1_scores))


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

def auroc(
    logits: np.ndarray,
    labels: np.ndarray,
    task: str = "vqacp",
) -> float:
    """Compute macro-averaged AUROC.

    Parameters
    ----------
    logits : ndarray (N, C)
        Raw model logits (pre-softmax/sigmoid).
    labels : ndarray (N,) or (N, C)
        True labels.
    task : str
        ``'vqacp'`` for multi-class, ``'mimic'`` for multi-label.

    Returns
    -------
    auc : float
        Macro-averaged AUROC.
    """
    try:
        if task in ("vqacp", "mimic"):
            # Multi-class: softmax over logits
            from scipy.special import softmax
            probs = softmax(logits, axis=1)
            if probs.shape[1] == 2:
                return float(roc_auc_score(labels, probs[:, 1]))
            else:
                return float(roc_auc_score(
                    labels, probs, multi_class="ovr", average="macro"
                ))
        else:
            raise ValueError(f"Unknown task: {task}")
    except ValueError:
        # AUROC undefined when some classes have no positive samples
        return 0.0


# ---------------------------------------------------------------------------
# Unified compute_all_metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(
    preds: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    groups: torch.Tensor | np.ndarray,
    quantum_outputs: Optional[torch.Tensor | np.ndarray] = None,
    task: str = "vqacp",
) -> dict[str, float]:
    """Compute all QEVC evaluation metrics in one call.

    Parameters
    ----------
    preds : Tensor or ndarray (N, C)
        Raw logits from the quantum head.
    labels : Tensor or ndarray
        Ground-truth labels.
    groups : Tensor or ndarray
        Bias group assignments.
    quantum_outputs : Tensor or ndarray (N, n_qubits), optional
        Raw quantum expectation values for QFS.
    task : str
        ``'vqacp'`` or ``'mimic'``.

    Returns
    -------
    metrics : dict
        Keys: ``accuracy``, ``eod``, ``ibd_f1``, ``auroc``, ``qfs``.
    """
    # Convert to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(groups, torch.Tensor):
        groups = groups.detach().cpu().numpy()

    logits = preds  # keep raw logits for AUROC

    # Hard predictions
    if task in ("vqacp", "mimic"):
        hard_preds = preds.argmax(axis=1)
        accuracy = float((hard_preds == labels).mean())
    else:
        raise ValueError(f"Unknown task: {task}")

    # Metrics
    eod_val = equalized_odds_difference(hard_preds, labels, groups)
    ibd_val = ibd_f1(hard_preds, labels, groups)
    auc_val = auroc(logits, labels, task=task)

    result = {
        "accuracy": accuracy,
        "eod": eod_val,
        "ibd_f1": ibd_val,
        "auroc": auc_val,
    }

    # QFS (requires quantum outputs)
    if quantum_outputs is not None:
        from qevc.quantum.qfs import quantum_feature_score
        qfs_val = quantum_feature_score(quantum_outputs, labels, groups)
        result["qfs"] = qfs_val

    return result
