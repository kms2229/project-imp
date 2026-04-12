"""PCA Fusion: reduce multi-modal embeddings to a compact feature vector.

Pipeline:
  1. Apply separate PCA to each modality (visual, language, structured)
  2. Concatenate the reduced representations
  3. Apply a final PCA to produce the ``n_pca``-dimensional fused vector

The fitted PCA models are saved as pickle files for reuse.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA


class PCAFusion:
    """Multi-modal PCA fusion for embedding dimensionality reduction.

    Parameters
    ----------
    n_pca : int
        Final fused feature dimension. Default: 32.
    n_components_per_modality : int or None
        Intermediate PCA dimension per modality before concatenation.
        If None, defaults to ``2 * n_pca``. With 2 modalities this gives
        ``4 * n_pca`` concat dims before the final PCA; with 3 modalities
        it gives ``6 * n_pca``.
    """

    def __init__(
        self,
        n_pca: int = 32,
        n_components_per_modality: Optional[int] = None,
    ) -> None:
        self.n_pca = n_pca
        self.n_per_mod = n_components_per_modality or (2 * n_pca)

        # Per-modality PCA objects (fitted during .fit())
        self.pca_visual: Optional[PCA] = None
        self.pca_language: Optional[PCA] = None
        self.pca_structured: Optional[PCA] = None
        # Final PCA on concatenated reduced embeddings
        self.pca_final: Optional[PCA] = None

        self._is_fitted = False

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(
        self,
        visual_embs: np.ndarray,
        language_embs: np.ndarray,
        structured_embs: Optional[np.ndarray] = None,
    ) -> "PCAFusion":
        """Fit all PCA stages on training embeddings.

        Parameters
        ----------
        visual_embs : ndarray (N, d_v)
            ViT CLS embeddings.
        language_embs : ndarray (N, d_l)
            RoBERTa CLS embeddings.
        structured_embs : ndarray (N, d_s) or None
            Structured MLP embeddings (MIMIC only).
        """
        print("Fitting PCA fusion...")

        # Fit per-modality PCA
        n_v = min(self.n_per_mod, visual_embs.shape[1])
        self.pca_visual = PCA(n_components=n_v, random_state=42)
        v_reduced = self.pca_visual.fit_transform(visual_embs)
        print(f"  Visual: {visual_embs.shape[1]} → {n_v}  "
              f"(explained var: {self.pca_visual.explained_variance_ratio_.sum():.3f})")

        n_l = min(self.n_per_mod, language_embs.shape[1])
        self.pca_language = PCA(n_components=n_l, random_state=42)
        l_reduced = self.pca_language.fit_transform(language_embs)
        print(f"  Language: {language_embs.shape[1]} → {n_l}  "
              f"(explained var: {self.pca_language.explained_variance_ratio_.sum():.3f})")

        parts = [v_reduced, l_reduced]

        if structured_embs is not None:
            n_s = min(self.n_per_mod, structured_embs.shape[1])
            self.pca_structured = PCA(n_components=n_s, random_state=42)
            s_reduced = self.pca_structured.fit_transform(structured_embs)
            print(f"  Structured: {structured_embs.shape[1]} → {n_s}  "
                  f"(explained var: {self.pca_structured.explained_variance_ratio_.sum():.3f})")
            parts.append(s_reduced)

        # Concatenate and apply final PCA
        concat = np.concatenate(parts, axis=1)
        n_final = min(self.n_pca, concat.shape[1])
        self.pca_final = PCA(n_components=n_final, random_state=42)
        self.pca_final.fit(concat)
        print(f"  Final: {concat.shape[1]} → {n_final}  "
              f"(explained var: {self.pca_final.explained_variance_ratio_.sum():.3f})")

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Transform
    # ------------------------------------------------------------------ #

    def transform(
        self,
        visual_embs: np.ndarray,
        language_embs: np.ndarray,
        structured_embs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transform embeddings using fitted PCA models.

        Returns
        -------
        ndarray (N, n_pca)
            Fused feature vectors.
        """
        if not self._is_fitted:
            raise RuntimeError("PCAFusion has not been fitted yet. Call .fit() first.")

        v_reduced = self.pca_visual.transform(visual_embs)
        l_reduced = self.pca_language.transform(language_embs)
        parts = [v_reduced, l_reduced]

        if structured_embs is not None and self.pca_structured is not None:
            s_reduced = self.pca_structured.transform(structured_embs)
            parts.append(s_reduced)

        concat = np.concatenate(parts, axis=1)
        return self.pca_final.transform(concat).astype(np.float32)

    def fit_transform(
        self,
        visual_embs: np.ndarray,
        language_embs: np.ndarray,
        structured_embs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(visual_embs, language_embs, structured_embs)
        return self.transform(visual_embs, language_embs, structured_embs)

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #

    def save(self, directory: str | Path) -> None:
        """Save fitted PCA models to pickle files.

        Creates: ``pca_v.pkl``, ``pca_l.pkl``, ``pca_s.pkl``, ``pca_final.pkl``
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / "pca_v.pkl", "wb") as f:
            pickle.dump(self.pca_visual, f)
        with open(directory / "pca_l.pkl", "wb") as f:
            pickle.dump(self.pca_language, f)
        with open(directory / "pca_final.pkl", "wb") as f:
            pickle.dump(self.pca_final, f)
        if self.pca_structured is not None:
            with open(directory / "pca_s.pkl", "wb") as f:
                pickle.dump(self.pca_structured, f)

        print(f"PCA models saved to {directory}")

    @classmethod
    def load(cls, directory: str | Path, n_pca: int = 32) -> "PCAFusion":
        """Load fitted PCA models from a directory."""
        directory = Path(directory)
        fusion = cls(n_pca=n_pca)

        with open(directory / "pca_v.pkl", "rb") as f:
            fusion.pca_visual = pickle.load(f)
        with open(directory / "pca_l.pkl", "rb") as f:
            fusion.pca_language = pickle.load(f)
        with open(directory / "pca_final.pkl", "rb") as f:
            fusion.pca_final = pickle.load(f)

        pca_s_path = directory / "pca_s.pkl"
        if pca_s_path.exists():
            with open(pca_s_path, "rb") as f:
                fusion.pca_structured = pickle.load(f)

        fusion._is_fitted = True
        print(f"PCA models loaded from {directory}")
        return fusion
