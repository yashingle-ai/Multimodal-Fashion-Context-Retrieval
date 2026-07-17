# Multimodal Fashion & Context Retrieval — Glance ML Assignment

Retrieve fashion images from a database using natural-language descriptions, understanding
**what** is worn, **where** the person is, and the **vibe** of the attire — going beyond a
vanilla CLIP baseline on compositionality and fine-grained fashion attributes.

Three independent, production-structured solutions are provided, each in its own directory
with a `src/` module (logic) separated from the data, and a runnable notebook that
executes the full **Indexer → Retriever → Evaluation** flow.

| | Approach | Fixes | Best at |
|---|----------|-------|---------|
| **1** | [Query Decomposition + Facet Scoring](approach1_query_decomposition/) | Compositionality / attribute binding | Interpretable multi-attribute queries |
| **2** | [Two-Stage Retrieve-and-Rerank](approach2_rerank/) | Context & precision via cross-encoder | Highest precision@k, scalable |
| **3** | [Cap-and-RAG (VLM captioning + text-to-text)](approach3_cap_and_rag/) ⭐ | Compositionality **and** context, via language | **Recommended** — strongest on complex/compositional queries |

Approaches 1–2 share a **FashionCLIP** backbone (CLIP fine-tuned on ~800K fashion pairs)
with **ChromaDB**. **Approach 3 (recommended)** bridges vision→language: a VLM (Qwen2-VL)
captions every image, captions are embedded with a text model (BGE), and retrieval is
text-to-text — where syntax and composition are handled natively.

## Repository layout
```
GLANCE assesment/
├── dataset/
│   ├── val_test2020.zip            # source (Fashionpedia val/test)
│   ├── test/                       # (generated) unzipped images
│   ├── images_1000/                # (generated) 1,000 sampled images — the DB
│   └── images_1000_manifest.json   # deterministic sample (seed=42)
├── approach1_query_decomposition/
├── approach2_rerank/
├── PROBLEM_ANALYSIS.md             # approaches, tradeoffs, chosen design, future work
└── README.md
```

## Setup (once)
The dataset is already extracted and sampled. To reproduce from the zip:
```bash
cd dataset && unzip -o val_test2020.zip
python3 - <<'PY'
import os, random, shutil
os.makedirs("images_1000", exist_ok=True)
files = sorted(f for f in os.listdir("test") if f.lower().endswith((".jpg",".jpeg",".png")))
random.seed(42)
for f in sorted(random.sample(files, 1000)):
    shutil.copy2(f"test/{f}", f"images_1000/{f}")
PY
```

## Run either approach
```bash
cd approach1_query_decomposition   # or approach2_rerank
pip install -r requirements.txt
jupyter lab                        # open the .ipynb and run top-to-bottom
```
First run downloads the pretrained models and builds the index (cached afterwards).
On Apple Silicon the code auto-selects the **MPS** GPU; otherwise CUDA, else CPU.

## The 5 evaluation queries (both notebooks run these)
1. "A person in a bright yellow raincoat."
2. "Professional business attire inside a modern office."
3. "Someone wearing a blue shirt sitting on a park bench."
4. "Casual weekend outfit for a city walk."
5. "A red tie and a white shirt in a formal setting."

Result grids are saved to each approach's `results/` folder.

## Where the ML thinking lives
See **[PROBLEM_ANALYSIS.md](PROBLEM_ANALYSIS.md)** for the full survey of approaches,
tradeoffs, the chosen architecture, and future work (adding locations/weather, improving
precision).
