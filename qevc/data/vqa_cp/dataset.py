"""VQA-CP v2 dataset loaders for the QEVC pipeline.

Two dataset classes:
  - VQACPRawDataset: loads raw annotations + image paths for embedding extraction
  - VQACPDataset: loads pre-extracted PCA-fused features for training/evaluation

Designed for BU SCC where HuggingFace-downloaded COCO images live in the
project disk cache, and annotations are stored as JSON.
"""

from __future__ import annotations

import json
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Paths for BU SCC ──
_SCC_PROJECT = Path("/projectnb/llm-plastsurg/kms_new")
_SCC_CACHE = _SCC_PROJECT / ".cache" / "huggingface" / "datasets" / "downloads" / "extracted"


def _find_coco_images_dir() -> Path:
    """Locate COCO images inside the HuggingFace cache on SCC.

    The HF datasets library extracts COCO zips into hash-named subdirs.
    We search for directories containing 'train2014' or 'val2014'.
    """
    # Search for val2014 directory (most likely present)
    candidates = list(_SCC_CACHE.glob("*/val2014"))
    if candidates:
        return candidates[0].parent

    # Fallback: search for any COCO jpg
    for d in _SCC_CACHE.iterdir():
        if d.is_dir():
            jpgs = list(d.glob("**/*.jpg"))
            if jpgs and "COCO" in jpgs[0].name:
                # Return the parent of train2014/val2014
                return jpgs[0].parent.parent

    raise FileNotFoundError(
        f"Could not find COCO images in {_SCC_CACHE}. "
        "Make sure Cell 6 (HuggingFace VQA download) has completed."
    )


def _build_image_id_to_path(coco_dir: Path) -> dict[int, str]:
    """Build a mapping from COCO image_id to file path.

    Scans both train2014/ and val2014/ directories.
    COCO filenames follow: COCO_{split}_{image_id:012d}.jpg
    """
    mapping = {}
    for split_dir in ["train2014", "val2014"]:
        img_dir = coco_dir / split_dir
        if not img_dir.exists():
            continue
        for jpg in img_dir.glob("*.jpg"):
            # Extract image_id from filename: COCO_val2014_000000201637.jpg
            try:
                image_id = int(jpg.stem.split("_")[-1])
                mapping[image_id] = str(jpg)
            except (ValueError, IndexError):
                continue

    if not mapping:
        raise FileNotFoundError(
            f"No COCO images found in {coco_dir}/train2014 or {coco_dir}/val2014"
        )
    return mapping


# ======================================================================== #
# VQACPRawDataset — for embedding extraction
# ======================================================================== #

class VQACPRawDataset:
    """Loads raw VQA-CP annotations and maps to COCO image paths.

    Used by ``extract_embeddings.py`` to get image paths + question texts
    before running through ViT and RoBERTa encoders.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``annotations_train.json`` and
        ``annotations_test.json`` (or the HuggingFace-format JSONs).
    split : str
        ``'train'`` or ``'test'``.
    max_answers : int
        Number of top answers to keep. Default: 3129 (standard VQA-CP).
    """

    # Question-type groups for bias analysis
    # Group 0: yes/no questions, Group 1: number, Group 2: other
    QTYPE_GROUPS = {
        "yes/no": 0,
        "number": 1,
        "other": 2,
    }

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        max_answers: int = 3129,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_answers = max_answers

        # Load annotations
        ann_path = self.data_dir / f"annotations_{split}.json"
        if not ann_path.exists():
            # Try the synth annotations as fallback
            ann_path = self.data_dir / f"annotations_{split}_synth.json"

        print(f"Loading annotations from {ann_path}...")
        with open(ann_path) as f:
            self.annotations = json.load(f)
        print(f"  Loaded {len(self.annotations):,} samples")

        # Build answer vocabulary from train split
        vocab_path = self.data_dir / "answer_vocab.json"
        if vocab_path.exists():
            with open(vocab_path) as f:
                self.answer_vocab = json.load(f)
        else:
            self.answer_vocab = self._build_answer_vocab()

        # Map COCO image_ids to file paths
        print("Locating COCO images...")
        try:
            coco_dir = _find_coco_images_dir()
            self.image_map = _build_image_id_to_path(coco_dir)
            print(f"  Found {len(self.image_map):,} COCO images")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            self.image_map = {}

    def _build_answer_vocab(self) -> dict[str, int]:
        """Build top-K answer vocabulary from training annotations."""
        # Only build from train data
        train_path = self.data_dir / "annotations_train.json"
        if train_path.exists():
            with open(train_path) as f:
                train_anns = json.load(f)
        else:
            train_anns = self.annotations

        # Count answer frequencies
        from collections import Counter
        answer_counts = Counter()
        for ann in train_anns:
            answers = ann.get("answers", [])
            if isinstance(answers, list) and len(answers) > 0:
                if isinstance(answers[0], dict):
                    # Format: [{"answer": "yes"}, ...]
                    for a in answers:
                        answer_counts[a.get("answer", "")] += 1
                elif isinstance(answers[0], str):
                    for a in answers:
                        answer_counts[a] += 1

        # Take top-K
        top_answers = [a for a, _ in answer_counts.most_common(self.max_answers)]
        vocab = {ans: idx for idx, ans in enumerate(top_answers)}

        # Save for reuse
        with open(self.data_dir / "answer_vocab.json", "w") as f:
            json.dump(vocab, f)
        print(f"  Built answer vocab: {len(vocab)} answers")

        return vocab

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample for embedding extraction.

        Returns
        -------
        dict with keys:
            - image_path: str (path to COCO image)
            - question: str
            - label: int (answer class index)
            - group: int (question-type group for bias)
        """
        ann = self.annotations[idx]

        # Image path
        image_id = ann.get("image_id", 0)
        image_path = self.image_map.get(image_id, "")

        # Question text
        question = ann.get("question", "")

        # Answer label (majority answer)
        answers = ann.get("answers", [])
        if isinstance(answers, list) and len(answers) > 0:
            if isinstance(answers[0], dict):
                answer_text = answers[0].get("answer", "")
            else:
                answer_text = str(answers[0])
        else:
            answer_text = ""
        label = self.answer_vocab.get(answer_text, 0)

        # Question-type group for bias analysis
        qtype = ann.get("question_type", "other")
        # Map question types to broad categories
        if qtype in ("yes/no",) or question.lower().startswith(("is ", "are ", "do ", "does ", "was ", "were ", "has ", "have ", "can ")):
            group = 0  # yes/no
        elif qtype in ("number",) or question.lower().startswith("how many"):
            group = 1  # number
        else:
            group = 2  # other

        return {
            "image_path": image_path,
            "question": question,
            "label": label,
            "group": group,
        }

    @property
    def n_classes(self) -> int:
        return len(self.answer_vocab)


# ======================================================================== #
# VQACPDataset — for training on pre-extracted PCA features
# ======================================================================== #

class VQACPDataset(Dataset):
    """PyTorch Dataset that loads pre-extracted PCA-fused features.

    Used by ``train_qevc.py`` and ``run_baselines.py`` after embeddings
    have been extracted and PCA-fused.

    Expects files in data_dir:
        - fused_{split}.npy     — (N, n_pca) float32 features
        - meta_{split}.npz      — labels (N,) int64, groups (N,) int64
        - answer_vocab.json      — answer vocabulary

    Parameters
    ----------
    data_dir : Path
        Directory with pre-extracted data.
    split : str
        ``'train'`` or ``'test'``.
    n_samples : int or None
        Limit dataset size (for sanity checks).
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        n_samples: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split

        # Load features
        fused_path = self.data_dir / f"fused_{split}.npy"
        meta_path = self.data_dir / f"meta_{split}.npz"

        self.features = np.load(fused_path).astype(np.float32)
        meta = np.load(meta_path)
        self.labels = meta["labels"].astype(np.int64)
        self.groups = meta["groups"].astype(np.int64)

        # Load vocab for n_classes
        vocab_path = self.data_dir / "answer_vocab.json"
        with open(vocab_path) as f:
            self._vocab = json.load(f)

        # Optional subsetting
        if n_samples is not None and n_samples < len(self.features):
            self.features = self.features[:n_samples]
            self.labels = self.labels[:n_samples]
            self.groups = self.groups[:n_samples]

        print(f"VQACPDataset({split}): {len(self)} samples, "
              f"{self.n_classes} classes, features={self.features.shape}")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": torch.from_numpy(self.features[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "group": torch.tensor(self.groups[idx], dtype=torch.long),
        }

    @property
    def n_classes(self) -> int:
        return len(self._vocab)
