"""BQA variational quantum circuit and the hybrid QEVC model.

Architecture per layer:
  1. **Data re-uploading** — angle encode n_qubits features via RY gates
  2. **Trainable rotations** — RY(w1) + RZ(w2) per qubit
  3. **Entangling** — circular CNOT chain (q_i → q_{i+1 mod n})

The QEVCModel wraps the circuit as a PyTorch module with parallel
classical and quantum classification heads.
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn

from .encoding import angle_encode, prepare_features


# ---------------------------------------------------------------------------
# Raw PennyLane circuit
# ---------------------------------------------------------------------------

def create_bqa_circuit(n_qubits: int, n_layers: int):
    """Create the BQA (Biased Question Answering) variational quantum circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers (each includes data re-uploading).

    Returns
    -------
    qnode : QNode
        PennyLane QNode with ``interface='torch'``.
        Uses ``lightning.qubit`` (C++ backend) with adjoint differentiation
        for faster simulation. Falls back to ``default.qubit`` if lightning
        is not installed.
    """
    # lightning.qubit is a C++ accelerated simulator — same quantum math,
    # ~3-5x faster than default.qubit's Python-based simulation.
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        diff_method = "adjoint"
    except qml.DeviceError:
        dev = qml.device("default.qubit", wires=n_qubits)
        diff_method = "backprop"
        print("WARNING: lightning.qubit not available, falling back to default.qubit")

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(features_padded, weights):
        """
        Parameters
        ----------
        features_padded : Tensor (n_layers * n_qubits,)
            Pre-processed features (arctan-scaled, zero-padded).
        weights : Tensor (n_layers, n_qubits, 2)
            Trainable rotation parameters [RY, RZ] per qubit per layer.
        """
        for layer in range(n_layers):
            # 1. Data re-uploading: angle encode this layer's features
            angle_encode(features_padded, n_qubits, layer_idx=layer)

            # 2. Trainable rotations
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)

            # 3. Entangling layer: circular CNOT chain
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])

        # Measure PauliZ expectation on all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


# ---------------------------------------------------------------------------
# Hybrid PyTorch model
# ---------------------------------------------------------------------------

class QEVCModel(nn.Module):
    """Hybrid classical-quantum model for QEVC.

    The model has two parallel paths:
      * **Quantum path**: PCA features → angle encoding → variational circuit
        → expectation values (n_qubits) → linear head → logits
      * **Classical path**: PCA features → MLP → logits

    During training, the hybrid loss combines both paths via λ weighting.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the quantum circuit.
    n_layers : int
        Number of variational layers.
    n_pca : int
        Dimension of input PCA-fused features.
    n_classes : int
        Number of output classes (3129 for VQA-CP, 50 for MIMIC top-50).
    """

    def __init__(
        self,
        n_qubits: int = 7,
        n_layers: int = 6,
        n_pca: int = 32,
        n_classes: int = 3129,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_pca = n_pca
        self.n_classes = n_classes

        # --- Quantum circuit ---
        self.circuit = create_bqa_circuit(n_qubits, n_layers)

        # Trainable quantum rotation parameters: (n_layers, n_qubits, 2)
        self.quantum_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 2) * 0.1
        )

        # Learnable feature scaling before angle encoding
        self.feature_scale = nn.Parameter(torch.ones(n_pca))

        # --- Quantum classification head ---
        self.quantum_head = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

        # --- Classical classification head (parallel path) ---
        self.classical_head = nn.Sequential(
            nn.Linear(n_pca, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through both quantum and classical paths.

        Parameters
        ----------
        x : Tensor (B, n_pca)
            PCA-fused input features.

        Returns
        -------
        q_logits : Tensor (B, n_classes)
            Quantum path output logits.
        c_logits : Tensor (B, n_classes)
            Classical path output logits.
        q_features : Tensor (B, n_qubits)
            Raw quantum expectation values (for QFS computation).
        """
        batch_size = x.shape[0]

        # Scale features
        x_scaled = x * self.feature_scale

        # --- Quantum path ---
        q_outputs = []
        for i in range(batch_size):
            # Prepare features for this sample
            features_padded = prepare_features(
                x_scaled[i], self.n_qubits, self.n_layers
            )
            # Run quantum circuit
            expvals = self.circuit(features_padded, self.quantum_weights)
            q_outputs.append(torch.stack(expvals))

        q_features = torch.stack(q_outputs).float()  # (B, n_qubits) — cast from float64
        q_logits = self.quantum_head(q_features)

        # --- Classical path ---
        c_logits = self.classical_head(x)

        return q_logits, c_logits, q_features

    @property
    def n_quantum_params(self) -> int:
        """Number of trainable quantum rotation parameters."""
        return self.n_layers * self.n_qubits * 2

    def quantum_state_summary(self) -> str:
        """Return a readable summary of the quantum circuit configuration."""
        return (
            f"QEVCModel: {self.n_qubits} qubits × {self.n_layers} layers = "
            f"{self.n_quantum_params} quantum params | "
            f"input={self.n_pca} → {self.n_classes} classes"
        )
