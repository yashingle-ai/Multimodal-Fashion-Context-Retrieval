"""Evaluation helpers: the 5 assessment queries + a result-grid visualizer."""
from __future__ import annotations

from pathlib import Path
from typing import List

from . import config

EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]


def show_results(query, results, cols: int = 5, save_as: str | None = None):
    """Render a titled grid of the retrieved images (for notebooks)."""
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(results)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.4 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, results):
        img = Image.open(config.IMAGE_DIR / r.filename).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"{r.score:.3f}", fontsize=9)
    fig.suptitle(query, fontsize=12, y=1.02)
    fig.tight_layout()
    if save_as:
        out = config.RESULTS_DIR / save_as
        fig.savefig(out, bbox_inches="tight", dpi=110)
        print(f"[eval] saved {out}")
    return fig


def run_all(retriever, k: int = 5, backend: str = "rule", save: bool = True):
    """Run every evaluation query and optionally save result grids."""
    all_results = {}
    for i, q in enumerate(EVAL_QUERIES, 1):
        res = retriever.search(q, k=k, backend=backend)
        all_results[q] = res
        if save:
            show_results(q, res, save_as=f"query_{i}.png")
    return all_results
