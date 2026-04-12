#!/usr/bin/env python3
"""Evaluate a trained QEVC checkpoint on the test set.

Usage:
    python -m qevc.scripts.evaluate --dataset vqacp --checkpoint qevc/checkpoints/qevc_vqacp_best.pt
    python -m qevc.scripts.evaluate --dataset mimic --checkpoint qevc/checkpoints/qevc_mimic_best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qevc.configs.config import load_config, set_seed, get_device
from qevc.quantum.circuit import QEVCModel
from qevc.evaluation.metrics import compute_all_metrics
from qevc.quantum.qfs import quantum_feature_score
from tqdm import tqdm


def evaluate(
    checkpoint_path: str,
    dataset: str,
    config_path: str,
    device_str: str = "auto",
):
    """Load checkpoint and run full evaluation."""

    device = get_device(device_str)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})

    # Load config (checkpoint config takes priority)
    config = load_config(config_path)
    n_qubits = ckpt_config.get("n_qubits", config.n_qubits)
    n_layers = ckpt_config.get("n_layers", config.n_layers)
    n_pca = ckpt_config.get("n_pca", config.n_pca)

    # Load test dataset
    if dataset == "vqacp":
        from qevc.data.vqa_cp.dataset import VQACPDataset
        data_dir = Path("qevc/data/vqa_cp")
        test_dataset = VQACPDataset(data_dir, split="test")
        n_classes = test_dataset.n_classes
    else:
        from qevc.data.mimic.dataset import MIMICDataset
        data_dir = Path("qevc/data/mimic")
        test_dataset = MIMICDataset(data_dir, split="test")
        n_classes = test_dataset.n_classes

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model and load weights
    model = QEVCModel(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_pca=n_pca,
        n_classes=n_classes,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(model.quantum_state_summary())
    print(f"Test set: {len(test_dataset)} samples\n")

    # --- Run evaluation ---
    all_q_logits = []
    all_labels = []
    all_groups = []
    all_q_features = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            features = batch["features"].to(device)
            labels = batch["label"]
            groups = batch["group"]

            q_logits, _, q_features = model(features)

            all_q_logits.append(q_logits.cpu())
            all_labels.append(labels)
            all_groups.append(groups)
            all_q_features.append(q_features.cpu())

    preds = torch.cat(all_q_logits)
    labels = torch.cat(all_labels)
    groups = torch.cat(all_groups)
    q_features = torch.cat(all_q_features)

    # Compute all metrics
    metrics = compute_all_metrics(
        preds=preds,
        labels=labels,
        groups=groups,
        quantum_outputs=q_features,
        task=dataset,
    )

    # --- Print results table ---
    print("\n" + "="*50)
    print("QEVC EVALUATION RESULTS")
    print("="*50)
    print(f"{'Metric':<20} {'Value':>10}")
    print("-"*30)
    for k, v in metrics.items():
        print(f"{k:<20} {v:>10.4f}")
    print("="*50)

    # --- Save results ---
    results_dir = Path("qevc/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"eval_{dataset}.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate QEVC checkpoint")
    parser.add_argument("--dataset", choices=["vqacp", "mimic"], required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pt checkpoint file",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    evaluate(args.checkpoint, args.dataset, args.config, args.device)


if __name__ == "__main__":
    main()
