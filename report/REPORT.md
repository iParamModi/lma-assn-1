# RAG Report: Baseline Retrieval vs. Cross-Encoder Re-Ranking

**Corpus:** Catan base-game rulebook (16 pages → 100 chunks, ~635 chars each).
**Bi-encoder:** `all-MiniLM-L6-v2` (384-dim), cosine similarity via a FAISS inner-product index.
**Cross-encoder:** `ms-marco-MiniLM-L-6-v2`.
**Answer model:** `Qwen/Qwen2.5-1.5B-Instruct`, run locally.
**Protocol:** the baseline retrieves the top 10 and sends its top 3 to the LLM; re-ranking re-scores the same 10 and sends its new top 3. The prompt is identical for both — only the three passages change.

---

## 1. Architecture Comparison

A **bi-encoder** encodes the query and each chunk *separately* into fixed vectors; relevance is the cosine similarity between two vectors that never met. Because the chunk vectors are computed once and stored in an index, search over the whole corpus is cheap (one query encode + a vector lookup). The weakness is precision: the two texts interact only through the geometry of the embedding space, so surface wording and topic overlap can outrank the passage that actually answers the question.

A **cross-encoder** concatenates the query and one chunk and runs them through the transformer *together*, so self-attention crosses every token of both. It outputs a single relevance score for that pair. This joint reading handles negation, qualifiers and specific conditions much better, but it needs one forward pass per `(query, chunk)` pair and builds no reusable index — so it cannot be run over the whole corpus, only over a short shortlist.

---

## 2. Qualitative & Rank-Shift Analysis

### Rank shifts (measured)

Chunks the cross-encoder lifted from ranks 4–10 into the top 3 — the passages the baseline would have withheld from the LLM:

| Query | Category | Chunk | Page | Baseline rank | Re-rank rank |
|---|---|---:|---:|---:|---:|
| What happens when a seven is rolled? | Keyword trap / distractor | 23 | 5 | 6 | 1 |
| What happens when a seven is rolled? | " | 63 | 11 | 8 | 2 |
| What happens when a seven is rolled? | " | 64 | 11 | 4 | 3 |
| Can I build a settlement right next to another settlement? | Exception & corner case | 67 | 12 | 8 | 1 |
| Can I give away resource cards to another player as a gift? | Negation / constraint | 37 | 7 | 5 | 1 |
| Can I give away resource cards to another player as a gift? | " | 79 | 14 | 7 | 2 |
| How many points do I need to win? | Buried specific | 39 | 7 | 4 | 1 |

### Answer 1 (baseline) vs Answer 2 (re-ranked)

- **Seven is rolled (keyword trap).** The baseline top 3 were the general resource-production text, the number-token passage that says there is "no 7", and a contents page — all high on wording, none describing the event. The discard-and-robber passages sat at ranks 4/6/8. **Answer 1** said only *"the robber token is activated"* and omitted the rest. **Answer 2** gave the full procedure: players holding more than 7 cards discard half (rounding down) and the robber blocks production. Clear win for re-ranking.

- **Give away cards (negation).** The query reads as "can I…", but the rule is a prohibition. The baseline surfaced generic trade text and **Answer 1 justified its reply with the wrong rule** (the resource-shortage clause) — a mild hallucination. The cross-encoder promoted the prohibition chunk from rank 5 to 1, and **Answer 2** stated it directly: *"You may not give away cards."* Clear win for re-ranking.

- **Settlement distance (exception).** The Distance-Rule example was buried at rank 8 → promoted to rank 1. Both answers reached "no, the Distance Rule applies", so the final answer quality was similar here even though re-ranking retrieved the governing passage much more cleanly.

- **Points to win (buried specific).** The worked example stating 10 points moved from rank 4 to 1. Both answers correctly said 10 — the number appears in several passages, so the baseline answered correctly despite not ranking the specific example first. The rank shift is real; its effect on the answer was negligible for this query.

**Takeaway:** re-ranking reliably pulls the governing passage into the top 3. Whether that changes the final answer depends on the question — it mattered most where the key fact lived in exactly one place (seven rolled, the give-away prohibition) and least where the fact was repeated across the document (points to win).

---

## 3. Latency Profiling (measured)

Retrieval-only timing (the LLM call is excluded — it is identical for both pipelines), averaged over repeats on CPU:

| Query | Baseline (ms) | Re-rank pipeline (ms) | Overhead (ms) | Slowdown |
|---|---:|---:|---:|---:|
| Seven is rolled | 23.1 | 774.1 | 751.0 | 33.5× |
| Settlement distance | 24.3 | 693.3 | 669.0 | 28.6× |
| Give away cards | 27.1 | 468.4 | 441.3 | 17.3× |
| Points to win | 22.7 | 497.6 | 474.8 | 21.9× |

**Reading:** baseline retrieval is a single query encode plus a vector lookup — tens of milliseconds. Re-ranking adds ~0.44–0.75 s because the cross-encoder runs 10 separate `(query, chunk)` forward passes on CPU. That is a ~17–34× slowdown on the retrieval step, but in absolute terms it is still under a second and is dwarfed by local LLM generation (~1 minute per answer on this machine).

---

## 4. Scaling Question

We do not run the cross-encoder across the whole collection from the start because it has no reusable index: relevance is only defined for a `(query, chunk)` pair read together, so scoring N chunks means N forward passes **per query**. On this 100-chunk corpus that is already ~0.5–0.75 s; on a realistic corpus of millions of chunks it would be minutes-to-hours per query — infeasible.

The bi-encoder embeds every chunk once (offline) into an index, so each query is one encode plus a sub-linear nearest-neighbour search. The standard pattern, used here, is therefore two stages: the cheap bi-encoder narrows the corpus to a short shortlist (top 10), and the expensive cross-encoder is spent only on those few candidates. This keeps most of the cross-encoder's precision at a tiny fraction of its cost.
