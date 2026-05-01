#!/usr/bin/env python3
"""Run classical baselines on the same PCA-fused features as QEVC.

Usage:
    python -m qevc.scripts.run_baselines --dataset vqacp --baselines all
    python -m qevc.scripts.run_baselines --dataset vqacp --baselines svm mlp
    python -m qevc.scripts.run_baselines --dataset mimic --baselines advdeb dba
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qevc.configs.config import load_config, set_seed, get_device
from qevc.evaluation.metrics import compute_all_metrics


AVAILABLE_BASELINES = ["svm", "mlp", "advdeb", "dba"]


def run_baselines(
    data_dir: Path,
    dataset: str,
    baselines: list[str],
    n_pca: int = 32,
):
    """Run specified baselines and evaluate."""

    device = get_device()

    # Load fused features
    X_train = np.load(data_dir / "fused_train.npy")
    meta_train = np.load(data_dir / "meta_train.npz", allow_pickle=True)

    if dataset in ("vqacp", "mimic"):
        y_train = meta_train["labels"].astype(np.int64)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    g_train = meta_train["groups"].astype(np.int64)

    # Test data
    X_test = np.load(data_dir / "fused_test.npy")
    meta_test = np.load(data_dir / "meta_test.npz", allow_pickle=True)
    if dataset in ("vqacp", "mimic"):
        y_test = meta_test["labels"].astype(np.int64)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    g_test = meta_test["groups"].astype(np.int64)

    n_classes = int(y_train.max()) + 1
    n_groups = int(g_train.max()) + 1

    print(f"Train: {X_train.shape} | Test: {X_test.shape} | "
          f"Classes: {n_classes} | Groups: {n_groups}")

    all_results = {}

    for name in baselines:
        print(f"\n{'='*50}")
        print(f"Baseline: {name.upper()}")
        print(f"{'='*50}")

        if name == "svm":
            from qevc.evaluation.baselines import SVMBaseline
            model = SVMBaseline(task=dataset)
            model.train(X_train, y_train)
            preds = model.predict(X_test)
            # SVM doesn't produce logits easily, use predictions for metrics
            metrics = {
                "accuracy": float((preds == y_test).mean()),
            }

        elif name == "mlp":
            from qevc.evaluation.baselines import MLPBaseline
            model = MLPBaseline(
                n_input=n_pca, n_classes=n_classes,
                task=dataset, device=device,
            )
            model.train(X_train, y_train)
            preds = model.predict(X_test)
            logits = model.predict_logits(X_test)
            metrics = compute_all_metrics(
                preds=logits, labels=y_test, groups=g_test, task=dataset,
            )

        elif name == "advdeb":
            from qevc.evaluation.baselines import AdvDebBaseline
            model = AdvDebBaseline(
                n_input=n_pca, n_classes=n_classes, n_groups=n_groups,
                task=dataset, device=device,
            )
            model.train(X_train, y_train, g_train)
            logits = model.predict_logits(X_test)
            metrics = compute_all_metrics(
                preds=logits, labels=y_test, groups=g_test, task=dataset,
            )

        elif name == "dba":
            from qevc.evaluation.baselines import DBABaseline
            model = DBABaseline(
                n_input=n_pca, n_classes=n_classes,
                task=dataset, device=device,
            )
            model.train(X_train, y_train, g_train)
            logits = model.predict_logits(X_test)
            metrics = compute_all_metrics(
                preds=logits, labels=y_test, groups=g_test, task=dataset,
            )
        else:
            print(f"  Unknown baseline: {name}")
            continue

        all_results[name] = metrics
        print(f"\n  Results for {name.upper()}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

    # Save all results
    results_dir = Path("qevc/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"baselines_{dataset}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ All results saved to {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run classical baselines")
    parser.add_argument("--dataset", choices=["vqacp", "mimic"], required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=AVAILABLE_BASELINES + ["all"],
        default=["all"],
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    baselines = AVAILABLE_BASELINES if "all" in args.baselines else args.baselines

    if args.dataset == "vqacp":
        data_dir = Path("qevc/data/vqa_cp")
    else:
        data_dir = Path("qevc/data/mimic")

    run_baselines(data_dir, args.dataset, baselines, n_pca=config.n_pca)


if __name__ == "__main__":
    main()
