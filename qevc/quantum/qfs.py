"""Quantum Feature Score (QFS).

Measures how well the quantum circuit's representations achieve
group-invariant feature learning. Higher QFS indicates the circuit
is learning features that generalize equally across bias groups,
which correlates with improved fairness (lower EOD).

QFS = 1 - (inter-group variance / total variance)
     × alignment factor between group centroids
"""

from __future__ import annotations

import numpy as np
import torch


def quantum_feature_score(
    quantum_outputs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    groups: torch.Tensor | np.ndarray,
) -> float:
    """Compute the Quantum Feature Score (QFS).

    QFS measures the degree to which quantum expectation values
    produce group-invariant representations — i.e., the quantum
    features look similar across different bias groups for the
    same true label.

    Parameters
    ----------
    quantum_outputs : Tensor or ndarray (N, n_qubits)
        Raw expectation values from the quantum circuit.
    labels : Tensor or ndarray (N,) or (N, n_codes)
        Ground-truth labels (int for single-label, multi-hot for multi-label).
    groups : Tensor or ndarray (N,)
        Bias group assignments (question type or demographic).

    Returns
    -------
    qfs : float
        Score in [0, 1]. Higher is better (more group-invariant).
    """
    # Convert to numpy
    if isinstance(quantum_outputs, torch.Tensor):
        quantum_outputs = quantum_outputs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(groups, torch.Tensor):
        groups = groups.detach().cpu().numpy()

    # For multi-label, use argmax of first label as proxy
    if labels.ndim == 2:
        labels = labels.argmax(axis=1)

    unique_groups = np.unique(groups)
    unique_labels = np.unique(labels)

    if len(unique_groups) < 2:
        return 1.0  # Only one group → trivially invariant

    # --- Compute per-group centroids within each label class ---
    total_variance = np.var(quantum_outputs, axis=0).mean()
    if total_variance < 1e-10:
        return 0.0

    inter_group_variances = []
    alignment_scores = []

    for label in unique_labels:
        label_mask = labels == label
        if label_mask.sum() < 2:
            continue

        group_centroids = []
        for group in unique_groups:
            mask = label_mask & (groups == group)
            if mask.sum() == 0:
                continue
            centroid = quantum_outputs[mask].mean(axis=0)
            group_centroids.append(centroid)

        if len(group_centroids) < 2:
            continue

        centroids = np.stack(group_centroids)  # (n_groups_present, n_qubits)

        # Inter-group variance for this label class
        inter_var = np.var(centroids, axis=0).mean()
        inter_group_variances.append(inter_var)

        # Alignment: cosine similarity between group centroids (pairwise average)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10
        normed = centroids / norms
        cos_sim = (normed @ normed.T)
        # Average off-diagonal
        n = len(centroids)
        if n > 1:
            mask_offdiag = ~np.eye(n, dtype=bool)
            align = cos_sim[mask_offdiag].mean()
            alignment_scores.append(max(0, align))

    if not inter_group_variances:
        return 0.5

    # QFS = (1 - inter_group_var / total_var) * alignment
    avg_inter_var = np.mean(inter_group_variances)
    variance_ratio = min(avg_inter_var / (total_variance + 1e-10), 1.0)
    invariance_score = 1.0 - variance_ratio

    avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.5

    qfs = invariance_score * avg_alignment
    return float(np.clip(qfs, 0.0, 1.0))
