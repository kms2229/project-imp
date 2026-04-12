"""Configuration loading and management for QEVC experiments.

Reads YAML config files and exposes typed dataclasses for use
throughout the pipeline. Also provides seed-setting and device
auto-detection utilities.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QEVCConfig:
    """Hyperparameters for a single QEVC training run."""

    n_qubits: int = 7
    n_layers: int = 6
    lam: float = 0.5          # called 'lambda' in YAML (Python keyword)
    lr: float = 0.01
    epochs: int = 50
    patience: int = 10
    n_pca: int = 32
    batch_size: int = 32
    seed: int = 42

    # Optional overrides (not in default YAML but useful for scripts)
    n_classes: int = 3129      # VQA-CP answer vocab size
    n_samples: Optional[int] = None   # limit dataset size for sanity checks
    dataset: str = "vqacp"     # 'vqacp' or 'mimic'
    device: str = "auto"       # 'auto', 'cpu', 'cuda', 'mps'


@dataclass
class AblationConfig:
    """Ranges for ablation study sweeps."""

    lambda_values: list[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0]
    )
    depth_values: list[int] = field(
        default_factory=lambda: [2, 4, 6, 8]
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_config(path: str | Path, **overrides) -> QEVCConfig:
    """Load a QEVCConfig from a YAML file, with optional keyword overrides.

    The YAML key ``lambda`` is mapped to the Python-safe field ``lam``.
    """
    with open(path) as f:
        raw: dict = yaml.safe_load(f) or {}

    # 'lambda' is a Python keyword → remap to 'lam'
    if "lambda" in raw:
        raw["lam"] = raw.pop("lambda")

    # Only pass keys that QEVCConfig knows about
    valid_keys = {f.name for f in QEVCConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    filtered.update(overrides)

    return QEVCConfig(**filtered)


def load_ablation_config(path: str | Path) -> AblationConfig:
    """Load an AblationConfig from a YAML file."""
    with open(path) as f:
        raw: dict = yaml.safe_load(f) or {}
    return AblationConfig(**raw)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across PyTorch, NumPy, and stdlib."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # PennyLane uses NumPy's RNG internally, so np.random.seed covers it.


def get_device(preference: str = "auto") -> torch.device:
    """Return the best available torch device.

    Parameters
    ----------
    preference : str
        One of ``'auto'``, ``'cpu'``, ``'cuda'``, ``'mps'``.
        ``'auto'`` selects CUDA > MPS > CPU in that priority order.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)
