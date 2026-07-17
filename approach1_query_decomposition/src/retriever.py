"""PART B — The Retriever (facet-scored, compositional).

Pipeline per query:
    1. Decompose the query into facets + (color->garment) bindings.
    2. Encode every facet phrase with FashionCLIP text encoder.
    3. Score all images against each facet (cosine, in the shared space).
    4. Fuse the per-facet scores with facet weights.
    5. Return top-k with a transparent per-facet score breakdown.

Why this beats vanilla CLIP: each binding phrase ("a red tie") is scored on its
own, so an image only wins if it matches the *bound* attribute, not merely the
set of words. Fusion lets scene/style contribute without drowning out the
garment evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .backbone import get_backbone
from .indexer import load_index
from .query_parser import DEFAULT_WEIGHTS, parse_query


@dataclass
class RetrievalResult:
    filename: str
    score: float
    facet_scores: Dict[str, float]


class FacetRetriever:
    def __init__(self, weights: Dict[str, float] | None = None):
        idx = load_index()
        self.ids: List[str] = idx["ids"]
        self.embeddings: np.ndarray = idx["embeddings"]        # (N, D), L2-normed
        self.backbone = get_backbone()
        self.weights = weights or DEFAULT_WEIGHTS

    def _facet_score(self, prompts: List[str]) -> np.ndarray:
        """Max cosine similarity of each image to any prompt in the facet.

        'max' means an image satisfying ANY phrasing of the facet scores high
        (robust to synonym choice); binding facets typically hold one phrase.
        """
        q = self.backbone.encode_texts(prompts)          # (P, D)
        sims = self.embeddings @ q.T                      # (N, P) cosine
        return sims.max(axis=1)                           # (N,)

    def search(self, query: str, k: int = 5, backend: str = "rule") -> List[RetrievalResult]:
        parsed = parse_query(query, backend=backend)
        facet_prompts = parsed.facet_prompts()

        # Per-facet image scores.
        facet_matrix, used_weights = {}, {}
        for facet, prompts in facet_prompts.items():
            facet_matrix[facet] = self._facet_score(prompts)
            used_weights[facet] = self.weights.get(facet, 0.3)

        # Weighted fusion (normalize weights so scores stay comparable).
        wsum = sum(used_weights.values()) or 1.0
        fused = np.zeros(len(self.ids), dtype="float32")
        for facet, scores in facet_matrix.items():
            fused += (used_weights[facet] / wsum) * scores

        top = np.argsort(-fused)[:k]
        return [
            RetrievalResult(
                filename=self.ids[i],
                score=float(fused[i]),
                facet_scores={f: float(facet_matrix[f][i]) for f in facet_matrix},
            )
            for i in top
        ]

    def explain(self, query: str, backend: str = "rule") -> dict:
        """Return the parsed facets — useful for demonstrating the decomposition."""
        return parse_query(query, backend=backend).as_dict()
