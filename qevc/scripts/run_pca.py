#!/usr/bin/env python3
"""Apply PCA fusion to pre-extracted embeddings.

Usage:
    python -m qevc.scripts.run_pca --dataset vqacp
    python -m qevc.scripts.run_pca --dataset mimic

Reads:   ev_{split}.npy, el_{split}.npy [, es_{split}.npy]
Writes:  fused_{split}.npy, pca_v.pkl, pca_l.pkl, pca_final.pkl [, pca_s.pkl]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qevc.configs.config import load_config, set_seed
from qevc.encoders.fusion import PCAFusion


def run_pca(data_dir: Path, n_pca: int = 32, has_structured: bool = False):
    """Fit PCA on train split and transform all splits."""

    # --- Load train embeddings ---
    print("Loading train embeddings...")
    ev_train = np.load(data_dir / "ev_train.npy")
    el_train = np.load(data_dir / "el_train.npy")
    es_train = None
    if has_structured:
        es_train = np.load(data_dir / "es_train.npy")

    print(f"  Visual:    {ev_train.shape}")
    print(f"  Language:  {el_train.shape}")
    if es_train is not None:
        print(f"  Structured: {es_train.shape}")

    # --- Fit PCA on train ---
    fusion = PCAFusion(n_pca=n_pca)
    fused_train = fusion.fit_transform(ev_train, el_train, es_train)
    np.save(data_dir / "fused_train.npy", fused_train)
    print(f"\n✓ fused_train.npy — shape {fused_train.shape}")

    # --- Transform test split ---
    ev_test_path = data_dir / "ev_test.npy"
    el_test_path = data_dir / "el_test.npy"

    if ev_test_path.exists() and el_test_path.exists():
        ev_test = np.load(ev_test_path)
        el_test = np.load(el_test_path)
        es_test = None
        if has_structured:
            es_test_path = data_dir / "es_test.npy"
            if es_test_path.exists():
                es_test = np.load(es_test_path)

        fused_test = fusion.transform(ev_test, el_test, es_test)
        np.save(data_dir / "fused_test.npy", fused_test)
        print(f"✓ fused_test.npy — shape {fused_test.shape}")
    else:
        print("⚠ Test embeddings not found — skipping test PCA transform")

    # --- Save fitted PCA models ---
    fusion.save(data_dir)

    print("\n✓ PCA fusion complete!")


def main():
    parser = argparse.ArgumentParser(description="Run PCA fusion on embeddings")
    parser.add_argument(
        "--dataset",
        choices=["vqacp", "mimic"],
        required=True,
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    if args.dataset == "vqacp":
        run_pca(Path("qevc/data/vqa_cp"), n_pca=config.n_pca, has_structured=False)
    else:
        run_pca(Path("qevc/data/mimic"), n_pca=config.n_pca, has_structured=True)


if __name__ == "__main__":
    main()
