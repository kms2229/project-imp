#!/usr/bin/env python3
"""Multi-GPU embedding extraction using ViT-L/14 and RoBERTa.

Uses torch.nn.DataParallel to distribute inference across all available GPUs.
Automatically scales batch size proportionally to the number of GPUs.

Usage:
    python -m qevc.scripts.extract_embeddings_multigpu --dataset vqacp
    python -m qevc.scripts.extract_embeddings_multigpu --dataset vqacp --batch-size 128

Outputs (same as single-GPU version):
    VQA-CP:  ev_{split}.npy, el_{split}.npy, meta_{split}.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from qevc.configs.config import set_seed, get_device


# ---------------------------------------------------------------------------
# Lightweight wrappers that work with DataParallel
# ---------------------------------------------------------------------------

class _ImagePathDataset(Dataset):
    """Loads images from file paths and applies a transform."""

    def __init__(self, image_paths: list[str], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.transform(img)


class _TextBatchDataset(Dataset):
    """Wraps a list of strings so DataLoader can batch them."""

    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class _ViTWrapper(nn.Module):
    """Thin wrapper around HuggingFace ViTModel for DataParallel."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :]  # CLS token


class _RoBERTaWrapper(nn.Module):
    """Thin wrapper around HuggingFace RobertaModel for DataParallel."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS token


# ---------------------------------------------------------------------------
# Multi-GPU extraction
# ---------------------------------------------------------------------------

def extract_vqacp_multigpu(
    data_dir: Path,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """Extract ViT + RoBERTa embeddings using all available GPUs.

    Parameters
    ----------
    data_dir : Path
        Directory containing annotations and where outputs will be saved.
    batch_size : int
        Per-process batch size. Effective batch = batch_size × n_gpus.
    num_workers : int
        DataLoader workers for image loading.
    """
    from qevc.data.vqa_cp.dataset import VQACPRawDataset

    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0")
    print(f"\n{'='*60}")
    print(f"  MULTI-GPU EXTRACTION: {n_gpus} GPU(s) detected")
    for i in range(n_gpus):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)")
    print(f"  Effective batch size: {batch_size * max(n_gpus, 1)}")
    print(f"{'='*60}\n")

    # --- Load ViT-L/14 ---
    print("Loading ViT-L/14...")
    from transformers import ViTModel, ViTImageProcessor
    vit_processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")
    vit_raw = ViTModel.from_pretrained("google/vit-large-patch16-224")
    vit_raw.eval()
    for p in vit_raw.parameters():
        p.requires_grad = False

    vit_wrapper = _ViTWrapper(vit_raw).to(device)
    if n_gpus > 1:
        vit_model = nn.DataParallel(vit_wrapper)
        print(f"  ViT wrapped with DataParallel across {n_gpus} GPUs")
    else:
        vit_model = vit_wrapper

    vit_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=vit_processor.image_mean, std=vit_processor.image_std),
    ])

    # --- Load RoBERTa ---
    print("Loading RoBERTa-base...")
    from transformers import RobertaModel, RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    roberta_raw = RobertaModel.from_pretrained("roberta-base")
    roberta_raw.eval()
    for p in roberta_raw.parameters():
        p.requires_grad = False

    roberta_wrapper = _RoBERTaWrapper(roberta_raw).to(device)
    if n_gpus > 1:
        roberta_model = nn.DataParallel(roberta_wrapper)
        print(f"  RoBERTa wrapped with DataParallel across {n_gpus} GPUs")
    else:
        roberta_model = roberta_wrapper

    # --- Process each split ---
    for split in ["train", "test"]:
        print(f"\n{'='*60}")
        print(f"  Processing VQA-CP {split} split")
        print(f"{'='*60}")

        ev_path = data_dir / f"ev_{split}.npy"
        el_path = data_dir / f"el_{split}.npy"
        meta_path = data_dir / f"meta_{split}.npz"

        # Load annotations
        raw_dataset = VQACPRawDataset(data_dir, split=split)

        # Collect all data
        print("  Collecting image paths, questions, labels, groups...")
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

        n_samples = len(image_paths)
        print(f"  Total samples: {n_samples:,}")

        # ── Visual embeddings ──
        if not ev_path.exists():
            print(f"\n  Extracting visual embeddings ({n_samples:,} images)...")
            img_dataset = _ImagePathDataset(image_paths, vit_transform)
            img_loader = DataLoader(
                img_dataset,
                batch_size=batch_size * max(n_gpus, 1),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )

            all_vis_embs = []
            t0 = time.time()
            with torch.no_grad():
                for batch_imgs in tqdm(img_loader, desc="  Visual"):
                    batch_imgs = batch_imgs.to(device)
                    emb = vit_model(batch_imgs)  # (B, 1024)
                    all_vis_embs.append(emb.cpu().numpy())

            ev = np.concatenate(all_vis_embs, axis=0)
            np.save(ev_path, ev)
            elapsed = time.time() - t0
            print(f"  ✓ Saved: {ev_path.name} — shape {ev.shape} — {elapsed:.0f}s "
                  f"({n_samples / elapsed:.0f} img/s)")
        else:
            print(f"  ✓ Visual embeddings already exist: {ev_path.name}")

        # ── Language embeddings ──
        if not el_path.exists():
            print(f"\n  Extracting language embeddings ({n_samples:,} questions)...")

            # Process in batches manually (tokenizer needs special handling)
            lang_batch_size = batch_size * max(n_gpus, 1) * 2  # text is faster
            all_lang_embs = []
            t0 = time.time()

            with torch.no_grad():
                for start in tqdm(range(0, n_samples, lang_batch_size), desc="  Language"):
                    end = min(start + lang_batch_size, n_samples)
                    batch_texts = questions[start:end]

                    tokens = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    input_ids = tokens["input_ids"].to(device)
                    attention_mask = tokens["attention_mask"].to(device)

                    emb = roberta_model(input_ids, attention_mask)  # (B, 768)
                    all_lang_embs.append(emb.cpu().numpy())

            el = np.concatenate(all_lang_embs, axis=0)
            np.save(el_path, el)
            elapsed = time.time() - t0
            print(f"  ✓ Saved: {el_path.name} — shape {el.shape} — {elapsed:.0f}s "
                  f"({n_samples / elapsed:.0f} q/s)")
        else:
            print(f"  ✓ Language embeddings already exist: {el_path.name}")

        # ── Metadata ──
        if not meta_path.exists():
            np.savez(
                meta_path,
                labels=np.array(labels, dtype=np.int64),
                groups=np.array(groups, dtype=np.int64),
            )
            print(f"  ✓ Saved: {meta_path.name}")
        else:
            print(f"  ✓ Metadata already exists: {meta_path.name}")

    # Free GPU memory
    del vit_model, roberta_model
    torch.cuda.empty_cache()
    print("\n✓ Multi-GPU extraction complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU embedding extraction for QEVC"
    )
    parser.add_argument(
        "--dataset", choices=["vqacp"], default="vqacp",
        help="Dataset to process (currently vqacp only)",
    )
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Per-GPU batch size (default: 64)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA GPUs. Use extract_embeddings.py for CPU.")

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")

    data_dir = Path("qevc/data/vqa_cp")
    extract_vqacp_multigpu(data_dir, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
