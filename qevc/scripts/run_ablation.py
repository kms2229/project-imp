#!/usr/bin/env python3
"""Run ablation studies over lambda and circuit depth.

Usage:
    # Run full ablation grid locally
    python -m qevc.scripts.run_ablation --dataset vqacp --param all

    # Run a single ablation point (for SLURM array jobs)
    python -m qevc.scripts.run_ablation --dataset vqacp --param lambda --value 0.25
    python -m qevc.scripts.run_ablation --dataset vqacp --param depth --value 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qevc.configs.config import load_config, load_ablation_config, set_seed


def run_single_ablation(dataset: str, config_path: str, param: str, value: float):
    """Run a single QEVC training with one ablation parameter override."""
    import subprocess
    import sys

    overrides = []
    if param == "lambda":
        overrides = ["--lam", str(value)]
        tag = f"lam_{value}"
    elif param == "depth":
        overrides = ["--n-layers", str(int(value))]
        tag = f"depth_{int(value)}"
    else:
        raise ValueError(f"Unknown ablation param: {param}")

    cmd = [
        sys.executable, "-m", "qevc.scripts.train_qevc",
        "--dataset", dataset,
        "--config", config_path,
        *overrides,
    ]

    print(f"\n{'='*50}")
    print(f"Ablation: {param} = {value}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def run_full_ablation(dataset: str, config_path: str, ablation_path: str):
    """Run all ablation combinations."""
    abl_config = load_ablation_config(ablation_path)

    results = {}

    # Lambda sweep (with default depth)
    print("\n" + "="*60)
    print("LAMBDA SWEEP")
    print("="*60)
    for lam in abl_config.lambda_values:
        rc = run_single_ablation(dataset, config_path, "lambda", lam)
        results[f"lambda_{lam}"] = {"returncode": rc}

    # Depth sweep (with default lambda)
    print("\n" + "="*60)
    print("DEPTH SWEEP")
    print("="*60)
    for depth in abl_config.depth_values:
        rc = run_single_ablation(dataset, config_path, "depth", depth)
        results[f"depth_{depth}"] = {"returncode": rc}

    # Save ablation summary
    results_dir = Path("qevc/results/ablation")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"ablation_summary_{dataset}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Ablation summary saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run QEVC ablation studies")
    parser.add_argument("--dataset", choices=["vqacp", "mimic"], default="vqacp")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ablation-config", default="configs/ablation.yaml")
    parser.add_argument(
        "--param",
        choices=["lambda", "depth", "all"],
        default="all",
        help="Which parameter to ablate",
    )
    parser.add_argument(
        "--value",
        type=float,
        default=None,
        help="Specific value (for single-point SLURM runs)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    if args.value is not None and args.param != "all":
        # Single-point run (used by SLURM array jobs)
        run_single_ablation(args.dataset, args.config, args.param, args.value)
    else:
        # Full grid
        run_full_ablation(args.dataset, args.config, args.ablation_config)


if __name__ == "__main__":
    main()
