# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Two retrieval pipelines over the same chunked corpus, compared head to head:

1. **Baseline** — dense bi-encoder retrieval (top 10 by cosine similarity), feed top 3 to the LLM.
2. **Re-ranking** — take the same top 10, re-score every `(query, chunk)` pair with a cross-encoder, feed the re-ranked top 3 to the LLM.

The project measures where the baseline alone fails and re-ranking recovers the right chunk, and profiles the latency cost of the extra stage.

## Corpus

A single complex PDF is used as the knowledge base: the **Catan base game rulebook** (rules + almanac). Rulebooks are a good stress test because they are full of exceptions, penalties, role limits and "except when…" clauses that reward precise retrieval.

The PDF is **not committed** (it is third-party content). The notebook downloads it to `data/` on first run. Source and the download URL are in the notebook's data cell.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

Models:
- Bi-encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Answer generation: `Qwen/Qwen2.5-1.5B-Instruct` run locally (override with the `LOCAL_MODEL` env var)

All models download from Hugging Face on first use and run on the local CPU. No API key or network service is needed — the entire notebook runs offline once the models are cached.

## Running

Open `rag_assignment.ipynb` and run top to bottom:

```bash
jupyter notebook rag_assignment.ipynb
```

## Layout

```
rag_assignment.ipynb   Main deliverable (data -> chunk -> index -> retrieve -> rerank -> analyze)
requirements.txt       Dependencies
data/                  Downloaded corpus (gitignored)
report/REPORT.md       Written report
```
