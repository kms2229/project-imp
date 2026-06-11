#!/usr/bin/env python3
"""Validate the Quantum Feature Score (QFS) metric against Equalized Odds Difference (EOD).
Requested by Reviewer 3: "How exactly do the proposed metrics (e.g., QFS) correlate
with standard bias metrics like EOD?"

Usage:
    python -m qevc.scripts.validate_qfs --dataset vqacp --checkpoints-dir qevc/checkpoints/ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path
import scipy.stats
import numpy as np

# We can reuse the evaluation logic
from qevc.scripts.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Validate QFS vs EOD")
    parser.add_argument("--dataset", choices=["vqacp", "mimic"], default="vqacp")
    parser.add_argument("--checkpoints-dir", default="qevc/checkpoints/ablation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoints_dir)
    if not ckpt_dir.exists():
        print(f"Error: Directory {ckpt_dir} does not exist.")
        print("Please run the ablation studies first to generate checkpoints.")
        return

    ckpts = list(ckpt_dir.glob("*.pt"))
    if not ckpts:
        print(f"No .pt checkpoints found in {ckpt_dir}.")
        # Fallback to the main checkpoint if no ablations exist yet
        fallback = Path(f"qevc/checkpoints/qevc_{args.dataset}_best.pt")
        if fallback.exists():
            print(f"Using fallback checkpoint: {fallback}")
            ckpts = [fallback]
        else:
            return

    results = []

    for ckpt in ckpts:
        print(f"\nEvaluating: {ckpt.name}")
        try:
            metrics = evaluate(
                checkpoint_path=str(ckpt),
                dataset=args.dataset,
                config_path=args.config,
                device_str=args.device
            )
            results.append({
                "name": ckpt.name,
                "qfs": metrics["qfs"],
                "eod": metrics["eod"],
                "accuracy": metrics["accuracy"],
                "ibd_f1": metrics["ibd_f1"]
            })
        except Exception as e:
            print(f"Failed to evaluate {ckpt.name}: {e}")

    if len(results) < 2:
        print("\nNeed at least 2 successful evaluations to compute correlation.")
        if len(results) == 1:
            print(f"Single result: QFS={results[0]['qfs']:.4f}, EOD={results[0]['eod']:.4f}")
        return

    qfs_vals = [r["qfs"] for r in results]
    eod_vals = [r["eod"] for r in results]

    # Compute Spearman rank correlation between QFS and EOD
    # Since QFS measures bias, and EOD measures bias, we expect a positive correlation.
    corr, p_value = scipy.stats.spearmanr(qfs_vals, eod_vals)

    print("\n" + "="*60)
    print(f"QFS vs EOD Validation ({args.dataset})")
    print("="*60)
    print(f"{'Model / Ablation':<30} | {'QFS':>8} | {'EOD':>8} | {'Accuracy':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<30} | {r['qfs']:>8.4f} | {r['eod']:>8.4f} | {r['accuracy']:>8.4f}")
    print("-" * 60)
    print(f"\nSpearman Correlation ρ(QFS, EOD): {corr:.4f} (p-value: {p_value:.4e})")
    print("="*60)
    print("Interpretation: A positive correlation indicates that as QFS (quantum")
    print("feature bias score) increases, the empirical fairness violation (EOD) also")
    print("increases, validating QFS as an effective internal bias indicator.")


if __name__ == "__main__":
    main()
