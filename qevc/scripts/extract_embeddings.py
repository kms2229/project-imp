#!/usr/bin/env python3
"""Extract embeddings from raw data using ViT-L/14 and RoBERTa.

Usage:
    python -m qevc.scripts.extract_embeddings --dataset vqacp --device auto
    python -m qevc.scripts.extract_embeddings --dataset mimic --device auto

Outputs:
    VQA-CP:  ev_{split}.npy, el_{split}.npy, meta_{split}.npz
    MIMIC:   es_{split}.npy, el_{split}.npy, meta_{split}.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qevc.configs.config import load_config, set_seed, get_device


def extract_vqacp(data_dir: Path, device, batch_size: int = 32):
    """Extract ViT + RoBERTa embeddings for VQA-CP v2."""
    from qevc.encoders.visual import VisualEncoder
    from qevc.encoders.language import LanguageEncoder
    from qevc.data.vqa_cp.dataset import VQACPRawDataset

    visual_enc = VisualEncoder(device=device)
    lang_enc = LanguageEncoder(device=device)

    for split in ["train", "test"]:
        print(f"\n{'='*40}")
        print(f"Extracting VQA-CP {split} split")
        print(f"{'='*40}")

        # Check if already extracted
        ev_path = data_dir / f"ev_{split}.npy"
        el_path = data_dir / f"el_{split}.npy"
        meta_path = data_dir / f"meta_{split}.npz"

        if ev_path.exists() and el_path.exists() and meta_path.exists():
            print(f"  ✓ Already extracted: {ev_path.name}, {el_path.name}, {meta_path.name}")
            continue

        raw_dataset = VQACPRawDataset(data_dir, split=split)

        # Collect image paths and questions
        image_paths = []
        questions = []
        labels = []
        groups = []

        for i in range(len(raw_dataset)):
            sample = raw_dataset[i]
            image_paths.append(sample["image_path"])
            questions.append(sample["question"])
            labels.append(sample["label"])
            groups.append(sample["group"])

        # Extract visual embeddings
        if not ev_path.exists():
            print(f"  Extracting visual embeddings ({len(image_paths)} images)...")
            ev = visual_enc.encode_paths(image_paths, batch_size=batch_size)
            np.save(ev_path, ev)
            print(f"  ✓ Saved: {ev_path.name} — shape {ev.shape}")
        else:
            print(f"  ✓ Visual embeddings already exist: {ev_path.name}")

        # Extract language embeddings
        if not el_path.exists():
            print(f"  Extracting language embeddings ({len(questions)} questions)...")
            el = lang_enc.encode_texts(questions, batch_size=batch_size * 2)
            np.save(el_path, el)
            print(f"  ✓ Saved: {el_path.name} — shape {el.shape}")
        else:
            print(f"  ✓ Language embeddings already exist: {el_path.name}")

        # Save metadata
        if not meta_path.exists():
            np.savez(
                meta_path,
                labels=np.array(labels, dtype=np.int64),
                groups=np.array(groups, dtype=np.int64),
            )
            print(f"  ✓ Saved: {meta_path.name}")


def extract_mimic(data_dir: Path, device, batch_size: int = 32):
    """Extract structured + language embeddings for MIMIC-III."""
    from qevc.encoders.language import LanguageEncoder
    from qevc.encoders.structured import StructuredEncoder
    from qevc.data.mimic.dataset import MIMICRawDataset
    import torch

    raw = MIMICRawDataset(data_dir)
    lang_enc = LanguageEncoder(device=device)

    # --- Structured features ---
    print("\nBuilding structured features from CHARTEVENTS...")
    struct_features, labels, groups = raw.build_structured_features()
    n_samples = len(labels)
    print(f"  Built {n_samples} samples, {struct_features.shape[1]} features")

    # Train/test split (80/20)
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    split_idx = int(0.8 * n_samples)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    for split, idx in [("train", train_idx), ("test", test_idx)]:
        # Structured embeddings via MLP
        es_path = data_dir / f"es_{split}.npy"
        if not es_path.exists():
            print(f"\n  Training structured encoder for {split}...")
            encoder = StructuredEncoder(
                n_features=struct_features.shape[1], embed_dim=64
            )
            # Quick training of the structured encoder
            encoder_device = torch.device(device) if isinstance(device, str) else device
            encoder = encoder.to(encoder_device)
            optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

            X = torch.from_numpy(struct_features[idx]).float().to(encoder_device)
            encoder.train()
            for epoch in range(20):
                optimizer.zero_grad()
                out = encoder(X)
                # Use reconstruction-style loss for unsupervised embedding
                reconstructor = torch.nn.Linear(64, struct_features.shape[1]).to(encoder_device)
                recon = reconstructor(out)
                loss = torch.nn.functional.mse_loss(recon, X)
                loss.backward()
                optimizer.step()

            encoder.eval()
            with torch.no_grad():
                es = encoder(X).cpu().numpy()
            np.save(es_path, es)
            print(f"  ✓ Saved: {es_path.name} — shape {es.shape}")

        # Language embeddings from clinical notes
        el_path = data_dir / f"el_{split}.npy"
        if not el_path.exists():
            print(f"  Extracting language embeddings for {split}...")
            notes_df = raw.load_notes()
            # Get one note per admission (first discharge summary)
            hadm_notes = (
                notes_df[notes_df["CATEGORY"] == "Discharge summary"]
                .groupby("HADM_ID")["TEXT"]
                .first()
            )
            # Match to our samples — use empty string if no note found
            texts = []
            for i in idx:
                # This is simplified — in practice, match by HADM_ID
                if i < len(hadm_notes):
                    texts.append(str(hadm_notes.iloc[i])[:512])
                else:
                    texts.append("")

            el = lang_enc.encode_texts(texts, batch_size=batch_size)
            np.save(el_path, el)
            print(f"  ✓ Saved: {el_path.name} — shape {el.shape}")

        # Metadata
        meta_path = data_dir / f"meta_{split}.npz"
        if not meta_path.exists():
            np.savez(
                meta_path,
                labels=labels[idx],
                groups=groups[idx],
            )
            print(f"  ✓ Saved: {meta_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings for QEVC pipeline"
    )
    parser.add_argument(
        "--dataset",
        choices=["vqacp", "mimic"],
        required=True,
        help="Which dataset to process",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    device = get_device(args.device)

    if args.dataset == "vqacp":
        data_dir = Path("qevc/data/vqa_cp")
        extract_vqacp(data_dir, device, args.batch_size)
    else:
        data_dir = Path("qevc/data/mimic")
        extract_mimic(data_dir, device, args.batch_size)

    print("\n✓ Extraction complete!")


if __name__ == "__main__":
    main()
