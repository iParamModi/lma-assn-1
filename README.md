# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Two retrieval pipelines over the same chunked corpus, compared head to head:

1. **Baseline** — dense bi-encoder retrieval (top 10 by cosine similarity), feed top 3 to the LLM.
2. **Re-ranking** — take the same top 10, re-score every `(query, chunk)` pair with a cross-encoder, feed the re-ranked top 3 to the LLM.

The project measures where the baseline alone fails and re-ranking recovers the right chunk, and profiles the latency cost of the extra stage.

## Corpus

A single complex PDF is used as the knowledge base: the **Catan base game rulebook** (rules + almanac). Rulebooks are a good stress test because they are full of exceptions, penalties, role limits and "except when…" clauses that reward precise retrieval.

The PDF is **not committed** (it is third-party content), so you need to place it yourself:

- **Download:** [Catan rulebook PDF (Google Drive)](https://drive.google.com/file/d/1BbUzyeNL1kCyzma_xemuMy7s-DwFdL1c/view?usp=sharing)
- **Save it as:** `data/catan_rules.pdf` (create the `data/` folder if it does not exist).

Alternatively, just run the notebook — its data cell downloads the same file to `data/catan_rules.pdf` automatically on first run if it is missing.

## Setup (reproduce from scratch)

**1. Get the code**

```bash
git clone https://github.com/iParamModi/lma-assn-1.git
cd lma-assn-1
```

**2. Create a virtual environment and install dependencies**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

**3. Get the corpus PDF** (see the *Corpus* section above)

```
data/catan_rules.pdf
```
Download it from the Drive link and place it there, or let the notebook fetch it automatically on first run.

**4. Run the notebook top to bottom**

```bash
jupyter notebook rag_assignment.ipynb
```

### Notes

- **Models** (all download from Hugging Face on first use, then run on the local CPU):
  - Bi-encoder: `sentence-transformers/all-MiniLM-L6-v2`
  - Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - Answer generation: `Qwen/Qwen2.5-1.5B-Instruct` (override with the `LOCAL_MODEL` env var)
- **No API key or paid service** is needed — everything runs locally, offline once the models are cached.
- First run downloads the models (~a few GB) and is slow on CPU; later runs reuse the cache.

## Layout

```
rag_assignment.ipynb   Main deliverable (data -> chunk -> index -> retrieve -> rerank -> analyze)
requirements.txt       Dependencies
data/                  Corpus PDF goes here
report/report.pdf      Written report
```
