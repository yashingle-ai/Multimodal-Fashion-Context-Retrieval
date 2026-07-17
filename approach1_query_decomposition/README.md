# Approach 1 — Query Decomposition + Facet Scoring

Text-to-image fashion retrieval that fixes CLIP's **compositionality** weakness by
decomposing the query into structured facets and `(color → garment)` bindings, scoring
each facet separately with **FashionCLIP**, and fusing the scores.

## Why it beats vanilla CLIP
Vanilla CLIP embeds a whole sentence into one vector (a "bag of concepts"), so
*"red tie + white shirt"* ≈ *"white tie + red shirt"*. Here, each binding phrase
("a red tie") is encoded and scored on its own, so the attribute stays bound to its
garment. Garment / environment / style are separate, weighted facets — which is what
makes multi-attribute queries work.

## Layout (logic separated from data)
```
src/
  config.py        # paths, model id, device selection
  backbone.py      # FashionCLIP wrapper (encode_images / encode_texts)
  indexer.py       # PART A: encode images -> ChromaDB + .npz cache
  query_parser.py  # query -> facets + bindings  (rule-based, optional LLM)
  retriever.py     # PART B: per-facet scoring + weighted fusion
  evaluate.py      # 5 assessment queries + result-grid visualizer
fashion_retrieval_approach1.ipynb   # run everything, see results
artifacts/         # (generated) persisted vector store + embedding cache
results/           # (generated) result grids per query
```
Data lives in `../dataset/images_1000/` — never inside the code.

## Run
```bash
pip install -r requirements.txt
jupyter lab fashion_retrieval_approach1.ipynb   # run top to bottom
```
Or from Python:
```python
from src.indexer import build_index; build_index()
from src.retriever import FacetRetriever
r = FacetRetriever()
r.search("A red tie and a white shirt in a formal setting.", k=5)
```

## Tuning
- Facet weights: `src/query_parser.py: DEFAULT_WEIGHTS`.
- Vocabulary (colors / garments / environments / styles): same file.
- LLM parser: `export OPENAI_API_KEY=...` then `search(..., backend="llm")`
  (falls back to rule-based automatically).

## Scalability to 1M images
Per-facet scoring is a single matmul at 1K images. At 1M, run each facet as its own
**ANN query** against the vector DB (top-M per facet) and merge/fuse the candidate
scores — cost stays sub-linear. The index build is a one-time offline batch job.
