"""Visual encoder using ViT-L/14 for image feature extraction.

Extracts CLS-token embeddings (dim=1024) from a frozen Vision Transformer.
Designed for batch processing of MS-COCO images on MPS/CUDA/CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image


class _ImagePathDataset(Dataset):
    """Minimal dataset that loads images from a list of file paths."""

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
            # Return a black image on failure (will be a zero embedding)
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.transform(img)


class VisualEncoder:
    """Frozen ViT-L/14 visual encoder for feature extraction.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: ``google/vit-large-patch14-224``.
    device : torch.device or str
        Target device. Default: auto-detected.
    """

    # ViT-L/14 output dimension
    EMBED_DIM = 1024

    def __init__(
        self,
        model_name: str = "google/vit-large-patch16-224",
        device: Optional[torch.device] = None,
    ) -> None:
        from transformers import ViTModel, ViTImageProcessor

        if device is None:
            from qevc.configs.config import get_device
            device = get_device()
        self.device = torch.device(device) if isinstance(device, str) else device

        print(f"Loading visual encoder: {model_name} → {self.device}")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # Standard torchvision transform (matches ViT expected input)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.processor.image_mean,
                std=self.processor.image_std,
            ),
        ])

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of pre-processed image tensors.

        Parameters
        ----------
        images : Tensor (B, 3, 224, 224)
            Batch of normalized image tensors.

        Returns
        -------
        Tensor (B, 1024)
            CLS token embeddings.
        """
        images = images.to(self.device)
        outputs = self.model(pixel_values=images)
        # CLS token is the first token in last_hidden_state
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb.cpu()

    @torch.no_grad()
    def encode_paths(
        self,
        image_paths: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Extract embeddings for a list of image file paths.

        Parameters
        ----------
        image_paths : list of str
            Paths to image files.
        batch_size : int
            Processing batch size (adjust for GPU memory).
        show_progress : bool
            Whether to show a tqdm progress bar.

        Returns
        -------
        ndarray (N, 1024)
            Stacked CLS embeddings for all images.
        """
        dataset = _ImagePathDataset(image_paths, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # safe for MPS
            pin_memory=self.device.type == "cuda",
        )

        all_embeddings = []
        iterator = tqdm(loader, desc="Visual encoding", disable=not show_progress)
        for batch in iterator:
            emb = self.encode(batch)
            all_embeddings.append(emb.numpy())

        return np.concatenate(all_embeddings, axis=0)
