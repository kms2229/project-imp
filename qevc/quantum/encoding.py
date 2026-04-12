"""Angle encoding: map classical features to qubit rotations.

Uses RY gates to encode real-valued features as rotation angles.
Features are bounded via arctan scaling to [-π/2, π/2].
"""

from __future__ import annotations

import pennylane as qml
import torch


def angle_encode(
    features: torch.Tensor,
    n_qubits: int,
    layer_idx: int = 0,
) -> None:
    """Apply angle encoding to qubits for one circuit layer.

    Encodes ``n_qubits`` features starting at position
    ``layer_idx * n_qubits`` from the feature vector.

    Parameters
    ----------
    features : Tensor (n_layers * n_qubits,)
        Pre-padded feature vector (arctan-scaled).
    n_qubits : int
        Number of qubits to encode onto.
    layer_idx : int
        Which layer's features to use (determines offset into features).
    """
    offset = layer_idx * n_qubits
    for q in range(n_qubits):
        qml.RY(features[offset + q], wires=q)


def prepare_features(
    features: torch.Tensor,
    n_qubits: int,
    n_layers: int,
) -> torch.Tensor:
    """Pre-process features for angle encoding.

    Applies arctan scaling and zero-pads to ``n_layers * n_qubits`` length.

    Parameters
    ----------
    features : Tensor (n_pca,)
        Raw PCA-fused feature vector.
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of circuit layers.

    Returns
    -------
    Tensor (n_layers * n_qubits,)
        Scaled and padded feature vector ready for the circuit.
    """
    total_slots = n_layers * n_qubits
    scaled = torch.arctan(features)

    padded = torch.zeros(total_slots, dtype=features.dtype, device=features.device)
    n_feat = min(len(scaled), total_slots)
    padded[:n_feat] = scaled[:n_feat]

    return padded
