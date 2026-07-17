"""PART A — The Indexer.

Turns raw images into FashionCLIP embeddings and persists them in a vector
store (ChromaDB). A parallel .npz cache holds the same vectors as a dense
matrix so the retriever can compute per-facet scores without re-encoding.

Run from a notebook / script:
    from src.indexer import build_index
    build_index()
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from . import config
from .backbone import get_backbone


def list_images(image_dir: Path = config.IMAGE_DIR) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)


def _get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def build_index(image_dir: Path = config.IMAGE_DIR, force: bool = False) -> dict:
    """Encode every image and persist to Chroma + an .npz matrix cache.

    Idempotent: if the cache already covers all images and ``force`` is False,
    it is reused instead of re-encoding.
    """
    paths = list_images(image_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    ids = [p.name for p in paths]

    if config.EMB_CACHE.exists() and not force:
        cached = np.load(config.EMB_CACHE, allow_pickle=True)
        if list(cached["ids"]) == ids:
            print(f"[indexer] cache hit: {len(ids)} embeddings reused.")
            return {"ids": ids, "embeddings": cached["embeddings"]}

    print(f"[indexer] encoding {len(paths)} images with FashionCLIP ...")
    backbone = get_backbone()
    embeddings = backbone.encode_images(paths).astype("float32")

    # Persist dense cache for fast facet scoring.
    np.savez(config.EMB_CACHE, ids=np.array(ids), embeddings=embeddings)

    # Persist to the vector DB (the assessment's "vector storage" requirement).
    col = _get_collection()
    if force or col.count() != len(ids):
        # rebuild cleanly
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass
        col = _get_collection()
        B = 512
        for i in range(0, len(ids), B):
            col.add(
                ids=ids[i : i + B],
                embeddings=embeddings[i : i + B].tolist(),
                metadatas=[{"filename": f} for f in ids[i : i + B]],
            )
    print(f"[indexer] done. {len(ids)} vectors in Chroma + {config.EMB_CACHE.name}.")
    return {"ids": ids, "embeddings": embeddings}


def load_index() -> dict:
    """Load the persisted embedding matrix (raises if the index is missing)."""
    if not config.EMB_CACHE.exists():
        raise FileNotFoundError("Index not built yet. Run build_index() first.")
    cached = np.load(config.EMB_CACHE, allow_pickle=True)
    return {"ids": list(cached["ids"]), "embeddings": cached["embeddings"]}
