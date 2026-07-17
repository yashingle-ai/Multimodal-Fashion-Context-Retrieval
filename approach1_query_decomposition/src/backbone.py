"""FashionCLIP backbone: a thin, cached wrapper around the HuggingFace model.

Exposes two operations used across the pipeline:
    - encode_images(paths) -> L2-normalized image embeddings
    - encode_texts(texts)  -> L2-normalized text embeddings

Both live in the same shared vector space, so cosine similarity between a text
vector and an image vector is a meaningful cross-modal relevance score.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .config import BACKBONE_MODEL, BATCH_SIZE, get_device


class FashionCLIP:
    """Encodes images and text into a shared, L2-normalized embedding space."""

    def __init__(self, model_name: str = BACKBONE_MODEL, device: str | None = None):
        self.device = device or get_device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @property
    def dim(self) -> int:
        return self.model.config.projection_dim

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            feats = self.model.get_text_features(**inputs)
            out.append(self._normalize(feats).cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def encode_images(self, paths: Iterable[Path | str]) -> np.ndarray:
        paths = [Path(p) for p in paths]
        out = []
        for i in range(0, len(paths), BATCH_SIZE):
            batch = paths[i : i + BATCH_SIZE]
            imgs = [Image.open(p).convert("RGB") for p in batch]
            inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            out.append(self._normalize(feats).cpu().numpy())
        return np.concatenate(out, axis=0)

    @staticmethod
    def _normalize(t) -> torch.Tensor:
        # Some transformers versions wrap features in a ModelOutput.
        if not torch.is_tensor(t):
            t = t.pooler_output if getattr(t, "pooler_output", None) is not None else t[0]
        return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@lru_cache(maxsize=1)
def get_backbone() -> FashionCLIP:
    """Process-wide singleton so the model is loaded only once."""
    return FashionCLIP()
