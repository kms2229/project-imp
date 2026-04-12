"""Structured encoder: 3-layer MLP for MIMIC-III tabular features.

Unlike the frozen ViT/RoBERTa encoders, this MLP is *trained* to learn
a compact representation of vital signs, lab values, and demographics.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StructuredEncoder(nn.Module):
    """Trainable 3-layer MLP encoder for structured clinical data.

    Architecture::

        Linear(n_features, 256) → ReLU → Dropout
        Linear(256, 128)        → ReLU → Dropout
        Linear(128, embed_dim)

    Parameters
    ----------
    n_features : int
        Number of input features (vital signs + labs + demographics).
    embed_dim : int
        Output embedding dimension. Default: 64.
    dropout : float
        Dropout probability between layers. Default: 0.2.
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
        )

        # Layer normalization on the output for stable fusion
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode structured features.

        Parameters
        ----------
        x : Tensor (B, n_features)
            Normalized structured input features.

        Returns
        -------
        Tensor (B, embed_dim)
            Learned structured embeddings.
        """
        return self.norm(self.encoder(x))

    @property
    def embed_dim(self) -> int:
        """Output embedding dimension."""
        return self.norm.normalized_shape[0]
