# Evaluation report — RAG over the Kubernetes documentation

Author: Hamza Ouba. Final-year project portfolio (Data & AI engineering).

All numbers in this report are generated from the result files by
`python -m ragk8s report` (blocks marked "pending" are not measured yet).

## 1. Goal

Build a retrieval-augmented question-answering system over the official
Kubernetes documentation and, above all, measure it: does it answer from its
sources, does it refuse when the answer is not there, and which chunking and
retrieval choices make a measurable difference once uncertainty is accounted for.

## 2. Corpus

- `content/en/docs/{concepts,tasks,tutorials}` of
  [kubernetes/website](https://github.com/kubernetes/website) at commit
  `5184b9b24b288709ed36b4003f53dc1530eda2f3` (2026-08-19): 408 pages after removing
  near-empty files. The snapshot in `data/raw/` is reproduced byte for byte by
  `python -m ragk8s ingest`; every page has a SHA-256 in the manifest.
- Hugo shortcodes and layout comments are rendered to plain text before chunking
  (e.g. glossary tooltips keep their text, version placeholders use the values of
  the pinned commit, feature-state markers without a version keep the gate name).

## 3. System

| Stage | Details |
|---|---|
| Chunking | fixed (1,000 chars, 150 overlap); structure (`##`/`###` sections, long sections re-split ≤ 1,500 chars with their heading); semantic (adjacent sentences merged while cosine similarity ≥ 0.55, ≤ 1,500 chars) |
| Indexes | BM25 (`rank_bm25`, lower-cased alphanumeric tokens); dense (`all-MiniLM-L6-v2`, normalised, FAISS inner product) |
| Retrieval | BM25; dense; hybrid = Reciprocal Rank Fusion (k = 60) of the top 60 of each; hybrid + rerank = cross-encoder `ms-marco-MiniLM-L-6-v2` on a candidate pool of 20,000 characters (20–80 chunks) |
| Context | chunks in rank order while they fit in 5,000 characters (max 20). Equal budget in characters, so strategies with small chunks are not starved |
| Prompt | numbered excerpts `[1]..[n]`; cite after each statement; exact refusal sentence when the answer is absent; English |
| Generators | application: `qwen/qwen3.8-27b` on Groq, `reasoning_effort="none"`, temperature 0, seed 42. Reference: `qwen2.5:7b-instruct` (Ollama, digest `845dbda0…`), temperature 0, seed 42, `num_ctx` 4096 |

## 4. Gold set

- **Factual questions: 45.** 55 questions were first drafted by `qwen2.5:7b` from
  single *structure* chunks; to avoid favouring that strategy, 25 more were drafted
  by `openai/gpt-oss-120b` from whole unused pages (seeded sample, proportional to
  sections); drafts whose evidence sentence was not found verbatim in the page were
  rejected automatically (3). All 77 remaining drafts were reviewed in one pass with
  three criteria — ground truth exact according to the page, question understandable
  on its own, a single correct answer — dropping rather than repairing doubtful
  ones. **Review process: verdicts proposed by an LLM assistant (Claude) against the
  source pages, then reviewed and validated without change by the author**
  (`reviewed_by` / `review_assisted_by` in `eval/questions.jsonl`). 32 were dropped,
  with reasons, in `eval/questions_dropped.jsonl`; 2 had a false premise.
- Kept: 27 chunk-based + 18 page-based ("neutral" subset).
- **Out-of-corpus questions: 8** (other technologies, non-existent API fields,
  false premises), used for the refusal rate.

## 5. Method

- **Retrieval metrics (no LLM).** A context is relevant when it contains a chunk of
  the question's source page (page level, so chunkings are comparable). Hit@1,
  Hit@3, Hit@ctx and MRR, on all questions and per origin subset.
- **Selection rule, fixed before any LLM evaluation:** the best configuration by MRR
  (`fixed + hybrid_rerank`) defines the 6 generated configurations: the 3 chunkings
  with that retriever, and that chunking with the 3 other retrievers.
- **RAGAS** (0.4.3) with judge `gemma4:12b-it-q8_0` (thinking off, temperature 0):
  faithfulness and answer relevancy on factual answers that are not refusals;
  context precision and recall on every factual question (shared by both
  generators, as they depend only on retrieval). Refusals are counted outside RAGAS
  with an exact sentence match: refusal rate on out-of-corpus questions, false
  refusal rate on factual ones.
- **Judge validation.** A seeded sample stratified by generator and by judge
  faithfulness, labelled blind by a human on three levels; Spearman correlation,
  MAE, judge-minus-human bias and Cohen's kappa.
- **Statistics.** 95% percentile bootstrap (10,000 resamples) over questions;
  comparisons use paired bootstrap differences on the questions scored in both.

## 6. Results

### 6.1 Retrieval

<!-- BEGIN:retrieval -->
Retrieval only, no LLM. 45 factual questions; relevant = the question's source page is in the
context given to the generator (5,000 characters for every strategy). Mean with 95% bootstrap CI. Computed 2026-11-22.

| Chunking | Retrieval | MRR | Hit@1 | Hit@ctx | ΔMRR vs best (paired) |
|---|---|---|---|---|---|
| fixed | hybrid_rerank | 0.84 [0.75, 0.92] | 0.76 | 0.96 | reference |
| structure | hybrid_rerank | 0.83 [0.73, 0.91] | 0.73 | 0.98 | -0.015 [-0.094, +0.059] |
| semantic | hybrid | 0.80 [0.70, 0.89] | 0.73 | 0.98 | -0.040 [-0.137, +0.045] |
| structure | hybrid | 0.79 [0.69, 0.87] | 0.67 | 0.96 | -0.056 [-0.160, +0.050] |
| fixed | hybrid | 0.77 [0.67, 0.87] | 0.67 | 0.96 | -0.067 [-0.172, +0.036] |
| semantic | hybrid_rerank | 0.76 [0.66, 0.86] | 0.67 | 0.98 | -0.077 [-0.157, -0.004] |
| semantic | bm25 | 0.76 [0.65, 0.86] | 0.67 | 0.96 | -0.083 [-0.193, +0.021] |
| structure | bm25 | 0.76 [0.65, 0.85] | 0.64 | 0.91 | -0.086 [-0.193, +0.017] |
| fixed | bm25 | 0.75 [0.64, 0.85] | 0.67 | 0.91 | -0.087 [-0.196, +0.016] |
| structure | dense | 0.68 [0.57, 0.79] | 0.58 | 0.89 | -0.159 [-0.279, -0.044] |
| semantic | dense | 0.66 [0.55, 0.77] | 0.51 | 0.96 | -0.182 [-0.285, -0.085] |
| fixed | dense | 0.63 [0.51, 0.75] | 0.51 | 0.84 | -0.207 [-0.337, -0.074] |
<!-- END:retrieval -->

Per-origin tables (neutral page subset vs chunk-based subset):
[`results/retrieval_metrics.md`](results/retrieval_metrics.md).

### 6.2 Generation and RAGAS

<!-- BEGIN:generation -->
_Pending: all runs judged by RAGAS, then `python -m ragk8s compare`._
<!-- END:generation -->

### 6.3 Application vs reference generator

<!-- BEGIN:models -->
_Pending: reference generator run fully judged, then `python -m ragk8s compare`._
<!-- END:models -->

### 6.4 Judge agreement with a human

<!-- BEGIN:agreement -->
_Pending: human labels (`python -m ragk8s validation export`, then `agree`)._
<!-- END:agreement -->

### 6.5 Run conditions

<!-- BEGIN:runs -->
| Model | Configuration | Provider | Model digest | Questions | Tokens / answer | s / answer | Code commit | Date (UTC) |
|---|---|---|---|---|---|---|---|---|
| qwen2.5:7b-instruct | fixed_hybrid_rerank | ollama | 845dbda0ea48 | 53 | 1342.8 | 4.61 | not recorded | 2026-11-22 13:16 |
| qwen/qwen3.8-27b | fixed_hybrid_rerank | groq | — | 53 | 1396.7 | 0.47 | 1368703 | 2026-11-22 11:58 |
| qwen/qwen3.8-27b | structure_hybrid_rerank | groq | — | 53 | 1438.7 | 0.5 | 1368703 (dirty) | 2026-11-22 12:10 |
<!-- END:runs -->

## 7. Limitations

- **Small gold set.** 45 factual questions (18 in the neutral subset): intervals are
  wide and only large differences are detectable.
- **LLM-assisted review.** The gold-set verdicts were proposed by an LLM and
  validated by the author, not written independently by a human.
- **Question origin.** 27 questions come from *structure* chunks; results are also
  reported on the page-based subset, which is small.
- **Single judge.** RAGAS scores depend on one 12B judge; its agreement with a human
  is measured on a small sample.
- **Preview model and free tier.** `qwen/qwen3.8-27b` is a preview model on Groq; its
  serving is not guaranteed to be deterministic or to stay available.
- **Traceability gaps, stated as they are.** The reference 7B run on Kaggle has no
  commit in its metadata (no git checkout there); it used commit `23ff591` recorded in
  the dataset manifest. One application run (`structure_hybrid_rerank`) is flagged
  `-dirty` because unrelated files (CLI, docs) were edited while it ran; the
  generation code was unchanged.
- **Corpus scope.** English concepts/tasks/tutorials only, frozen at one commit.

## 8. Reproducibility

Every command is in [docs/REPRODUCE.md](docs/REPRODUCE.md). The corpus, chunking,
indexes and retrieval are deterministic for a given environment; answers and judge
scores are cached by everything that determines them (model digest, prompt,
options, inputs), and each run records its conditions.
