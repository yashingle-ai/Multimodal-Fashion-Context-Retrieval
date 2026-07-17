"""Central configuration for Approach 1 (Query Decomposition + Facet Scoring).

All paths are resolved relative to the repository root so the code runs
identically from a notebook, a script, or a test.
"""
from __future__ import annotations

from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# .../GLANCE assesment/approach1_query_decomposition/src/config.py
SRC_DIR = Path(__file__).resolve().parent
APPROACH_DIR = SRC_DIR.parent
REPO_ROOT = APPROACH_DIR.parent

IMAGE_DIR = REPO_ROOT / "dataset" / "images_1000"
INDEX_DIR = APPROACH_DIR / "artifacts"          # persisted vector store + caches
CHROMA_DIR = INDEX_DIR / "chroma"
EMB_CACHE = INDEX_DIR / "image_embeddings.npz"  # numpy cache for fast facet scoring
RESULTS_DIR = APPROACH_DIR / "results"

COLLECTION_NAME = "fashion_fashionclip"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# FashionCLIP: a CLIP model fine-tuned on ~800K fashion image-text pairs.
# Drop-in replacement for openai/clip-vit-base-patch32 with much stronger
# fine-grained fashion understanding.
BACKBONE_MODEL = "patrickjohncyh/fashion-clip"

BATCH_SIZE = 32


def get_device() -> str:
    """Pick the best available device (Apple MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


for _d in (INDEX_DIR, CHROMA_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
