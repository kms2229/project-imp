"""Language encoder using RoBERTa-base for text feature extraction.

Extracts [CLS] pooled embeddings (dim=768) from a frozen RoBERTa model.
Handles both short VQA questions and longer MIMIC clinical notes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from tqdm import tqdm


class LanguageEncoder:
    """Frozen RoBERTa-base language encoder for feature extraction.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: ``roberta-base``.
    max_length : int
        Maximum token length (truncates longer inputs). Default: 512.
    device : torch.device or str
        Target device. Default: auto-detected.
    """

    # RoBERTa-base output dimension
    EMBED_DIM = 768

    def __init__(
        self,
        model_name: str = "roberta-base",
        max_length: int = 512,
        device: Optional[torch.device] = None,
    ) -> None:
        from transformers import RobertaModel, RobertaTokenizerFast

        if device is None:
            from qevc.configs.config import get_device
            device = get_device()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_length = max_length

        print(f"Loading language encoder: {model_name} → {self.device}")
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of text strings.

        Parameters
        ----------
        texts : list of str
            Batch of input texts (questions or clinical notes).

        Returns
        -------
        Tensor (B, 768)
            Pooled [CLS] embeddings.
        """
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        outputs = self.model(**tokens)
        # Use CLS token (index 0) from last hidden state
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb.cpu()

    @torch.no_grad()
    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Extract embeddings for a list of text strings.

        Parameters
        ----------
        texts : list of str
            All input texts.
        batch_size : int
            Processing batch size.
        show_progress : bool
            Whether to show a tqdm progress bar.

        Returns
        -------
        ndarray (N, 768)
            Stacked CLS embeddings for all texts.
        """
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Language encoding")

        for start in iterator:
            batch = texts[start : start + batch_size]
            emb = self.encode(batch)
            all_embeddings.append(emb.numpy())

        return np.concatenate(all_embeddings, axis=0)
