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
- **Judge agreement check.** A seeded sample of 40 answers, stratified by generator
  and by judge faithfulness, labelled on three levels (0 / 0.5 / 1) without access
  to the judge's scores; Spearman correlation, MAE, judge-minus-annotator bias and
  Cohen's kappa. A first human pass was discarded by its author as unreliable. The
  reported sample is a second, non-overlapping draw (the first 40 items excluded)
  labelled by an LLM (`claude-opus-5-5`), so it measures agreement between two
  models, not a human validation.
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
Generator `qwen3.8-27b`; judge `gemma4:12b-it-q8_0`. Mean with 95% bootstrap CI over questions.

| Chunking | Retrieval | Faithfulness | Answer relevancy | Context precision | Context recall | Refused out-of-corpus | False refusals |
|---|---|---|---|---|---|---|---|
| fixed | hybrid_rerank | 0.98 [0.95, 1.00] | 0.88 [0.83, 0.91] | 0.89 [0.82, 0.95] | 0.96 [0.89, 1.00] | 1.00 [1.00, 1.00] | 0.02 [0.00, 0.07] |
| structure | hybrid_rerank | 0.99 [0.98, 1.00] | 0.87 [0.83, 0.91] | 0.92 [0.87, 0.96] | 0.97 [0.91, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| semantic | hybrid_rerank | 0.98 [0.96, 1.00] | 0.90 [0.86, 0.93] | 0.72 [0.63, 0.81] | 0.90 [0.81, 0.98] | 1.00 [1.00, 1.00] | 0.04 [0.00, 0.11] |
| fixed | bm25 | 0.99 [0.98, 1.00] | 0.89 [0.85, 0.92] | 0.80 [0.70, 0.88] | 0.94 [0.88, 1.00] | 1.00 [1.00, 1.00] | 0.04 [0.00, 0.11] |
| fixed | dense | 0.99 [0.96, 1.00] | 0.88 [0.84, 0.92] | 0.65 [0.56, 0.74] | 0.93 [0.84, 1.00] | 1.00 [1.00, 1.00] | 0.07 [0.00, 0.16] |
| fixed | hybrid | 0.98 [0.95, 1.00] | 0.87 [0.83, 0.91] | 0.84 [0.76, 0.91] | 0.96 [0.89, 1.00] | 1.00 [1.00, 1.00] | 0.04 [0.00, 0.11] |

Full tables and paired differences: [`results/comparison.md`](results/comparison.md).
<!-- END:generation -->

### 6.3 Application vs reference generator

<!-- BEGIN:models -->
Same contexts (`fixed_hybrid_rerank`), same prompt, same judge. Δ = paired bootstrap difference (* = CI excludes 0).

| Metric | qwen3.8-27b | qwen2.5-7b-instruct | Δ |
|---|---|---|---|
| faithfulness | 0.98 [0.95, 1.00] | 0.94 [0.89, 0.99] | +0.039 [-0.008, +0.093] |
| answer relevancy | 0.88 [0.83, 0.91] | 0.86 [0.80, 0.91] | -0.012 [-0.063, +0.036] |
| citation coverage | 0.91 [0.84, 0.97] | 0.68 [0.56, 0.78] | +0.229 [+0.129, +0.338] * |
| seconds per answer (mean) | 0.47 | 4.61 | — |
<!-- END:models -->

### 6.4 Judge agreement check

<!-- BEGIN:agreement -->
40 answers labelled by an LLM (claude-opus-5-5, 0 / 0.5 / 1) without seeing the judge's scores: this is agreement between two models, not a human validation. Bias > 0 = judge more lenient than the annotator. n/a = one side gives the same value to every answer.

| Metric | Spearman [95% CI] | MAE | Bias (judge - annotator) | Cohen's κ (≥ 0.5) | Mean, annotator / judge |
|---|---|---|---|---|---|
| faithfulness | -0.04 [-0.09, -0.03] | 0.04 | -0.01 [-0.05, +0.03] | n/a | 0.99 / 0.97 |
| answer_relevancy | 0.27 [0.15, 0.46] | 0.13 | -0.11 [-0.15, -0.07] | n/a | 0.97 / 0.87 |
| context_recall | 0.72 [0.52, 1.00] | 0.01 | +0.01 [+0.00, +0.04] | 1.00 | 0.96 / 0.97 |
<!-- END:agreement -->

Reading: the levels agree (MAE ≤ 0.13, faithfulness bias within ±0.05), but the
judge's ability to *rank* answers cannot be tested on this sample: almost every
answer is faithful and relevant (ceiling effect), so a rank correlation has very
little variance to work with. The judge appears stricter than the annotator on
answer relevancy (bias −0.11). Faithfulness and relevancy numbers in 6.2 should be
read as levels, not as evidence that the judge separates good from bad answers.

### 6.5 Run conditions

<!-- BEGIN:runs -->
| Model | Configuration | Provider | Model digest | Questions | Tokens / answer | s / answer | Code commit | Date (UTC) |
|---|---|---|---|---|---|---|---|---|
| qwen2.5:7b-instruct | fixed_hybrid_rerank | ollama | 845dbda0ea48 | 53 | 1342.8 | 4.61 | not recorded | 2026-11-22 13:16 |
| qwen/qwen3.8-27b | fixed_bm25 | groq | — | 53 | 1428.8 | 0.51 | 6a96644 | 2026-11-24 13:08 |
| qwen/qwen3.8-27b | fixed_dense | groq | — | 53 | 1451.2 | 0.51 | 6a96644 | 2026-11-24 13:08 |
| qwen/qwen3.8-27b | fixed_hybrid | groq | — | 53 | 1433.8 | 0.5 | 6a96644 | 2026-11-24 13:22 |
| qwen/qwen3.8-27b | fixed_hybrid_rerank | groq | — | 53 | 1396.7 | 0.47 | 6a96644 | 2026-11-24 13:08 |
| qwen/qwen3.8-27b | semantic_hybrid_rerank | groq | — | 53 | 1439.6 | 0.48 | 6a96644 | 2026-11-24 13:08 |
| qwen/qwen3.8-27b | structure_hybrid_rerank | groq | — | 53 | 1438.7 | 0.5 | 6a96644 | 2026-11-24 13:08 |
<!-- END:runs -->

## 7. Limitations

- **Small gold set.** 45 factual questions (18 in the neutral subset): intervals are
  wide and only large differences are detectable.
- **LLM-assisted review.** The gold-set verdicts were proposed by an LLM and
  validated by the author, not written independently by a human.
- **Question origin.** 27 questions come from *structure* chunks; results are also
  reported on the page-based subset, which is small.
- **Single, unvalidated judge.** RAGAS scores depend on one 12B judge. No human
  validation is reported: the first human pass was discarded, and the agreement in
  6.4 is with another LLM, on 40 answers dominated by a ceiling effect.
- **Preview model and free tier.** `qwen/qwen3.8-27b` is a preview model on Groq; its
  serving is not guaranteed to be deterministic or to stay available.
- **Traceability gaps, stated as they are.** The reference 7B run on Kaggle has no
  commit in its metadata (no git checkout there); it used commit `23ff591` recorded in
  the dataset manifest. All six 27B runs are recorded at the clean commit `6a96644`;
  their answers were served from the run cache, whose key covers every input
  (model, prompt, contexts, decoding parameters).
- **Application trade-off.** The live app serves `fixed_hybrid` (no reranker: the
  cross-encoder takes about 5–10 s per query on one vCPU). Its MRR gap with the best
  configuration is −0.066 [−0.172, +0.036], not significant. Reranked results are
  shown only for evaluation questions, precomputed. The embedder runs in ONNX fp32
  (exact parity with PyTorch; int8 drifted).
- **Corpus scope.** English concepts/tasks/tutorials only, frozen at one commit.

## 8. Reproducibility

Every command is in [docs/REPRODUCE.md](docs/REPRODUCE.md). The corpus, chunking,
indexes and retrieval are deterministic for a given environment; answers and judge
scores are cached by everything that determines them (model digest, prompt,
options, inputs), and each run records its conditions.
