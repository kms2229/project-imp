#!/usr/bin/env python3
"""Train the QEVC hybrid quantum-classical model.

Usage:
    # Full training
    python -m qevc.scripts.train_qevc --dataset vqacp --config configs/default.yaml

    # Sanity check (500 samples, 5 epochs)
    python -m qevc.scripts.train_qevc --dataset vqacp --n-samples 500 --epochs 5

    # MIMIC
    python -m qevc.scripts.train_qevc --dataset mimic --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from qevc.configs.config import load_config, set_seed, get_device
from qevc.quantum.circuit import QEVCModel
from qevc.training.trainer import QEVCTrainer


def main():
    parser = argparse.ArgumentParser(description="Train QEVC model")
    parser.add_argument("--dataset", choices=["vqacp", "mimic"], default="vqacp")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit dataset size (for sanity checks)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lam", type=float, default=None,
                        help="Override lambda")
    parser.add_argument("--n-layers", type=int, default=None,
                        help="Override circuit depth")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (larger = faster epochs)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # --- Load config with overrides ---
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.lam is not None:
        overrides["lam"] = args.lam
    if args.n_layers is not None:
        overrides["n_layers"] = args.n_layers
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size

    config = load_config(args.config, **overrides)
    set_seed(config.seed)
    device = get_device(args.device)

    n_samples = args.n_samples or config.n_samples

    # --- Load dataset ---
    if args.dataset == "vqacp":
        from qevc.data.vqa_cp.dataset import VQACPDataset
        data_dir = Path("qevc/data/vqa_cp")
        full_dataset = VQACPDataset(data_dir, split="train", n_samples=n_samples)
        n_classes = full_dataset.n_classes
    else:
        from qevc.data.mimic.dataset import MIMICDataset
        data_dir = Path("qevc/data/mimic")
        full_dataset = MIMICDataset(data_dir, split="train", n_samples=n_samples)
        n_classes = full_dataset.n_classes

    # --- Train/val split (90/10) ---
    n_total = len(full_dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Dataset: {args.dataset} | Train: {n_train} | Val: {n_val} | "
          f"Classes: {n_classes}")

    # --- Create model ---
    model = QEVCModel(
        n_qubits=config.n_qubits,
        n_layers=config.n_layers,
        n_pca=config.n_pca,
        n_classes=n_classes,
    )
    print(model.quantum_state_summary())
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Train ---
    trainer = QEVCTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        task=args.dataset,
    )

    history = trainer.train()

    # --- Evaluate on test set if available ---
    if args.dataset == "vqacp":
        test_path = data_dir / "fused_test.npy"
    else:
        test_path = data_dir / "fused_test.npy"

    if test_path.exists():
        print("\nRunning evaluation on test set...")
        if args.dataset == "vqacp":
            test_dataset = VQACPDataset(data_dir, split="test")
        else:
            test_dataset = MIMICDataset(data_dir, split="test")

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        trainer.evaluate(test_loader)
    else:
        print("\n⚠ Test data not found — skipping evaluation")


if __name__ == "__main__":
    main()
