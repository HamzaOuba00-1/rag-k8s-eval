# rag-k8s-eval

**Does a RAG system tell the truth, and which design choices actually matter?**
A retrieval-augmented question-answering system over the official Kubernetes
documentation, built around its *evaluation*: every design choice (chunking,
retrieval, generator) is compared with measured numbers and 95% confidence
intervals, not by eye.

[![ci](https://github.com/HamzaOuba00-1/rag-k8s-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/HamzaOuba00-1/rag-k8s-eval/actions/workflows/ci.yml)

## What is measured

- **Retrieval** — 3 chunking strategies (fixed-size, Markdown structure, semantic) ×
  4 retrievers (BM25, dense, hybrid RRF, hybrid + cross-encoder reranking), scored
  without any LLM: is the question's source page in the context, and at which rank?
- **Generation** — answers must cite the excerpts they use (`[n]`) and refuse
  explicitly when the information is not there. Measured: RAGAS faithfulness,
  answer relevancy, context precision and recall; refusal rate on out-of-corpus
  questions; false refusals; share of sentences carrying a citation.
- **The judge itself** — RAGAS scores come from an LLM judge, so its agreement with
  a human is measured on a blind sample before its numbers are trusted.
- **Uncertainty** — about 50 questions is small: every mean has a bootstrap
  interval and every comparison is a *paired* bootstrap difference on the same
  questions.

## Results

### Retrieval (top configurations)

<!-- BEGIN:retrieval-top -->
Retrieval only, no LLM. 45 factual questions; relevant = the question's source page is in the
context given to the generator (5,000 characters for every strategy). Mean with 95% bootstrap CI. Computed 2026-11-22.

| Chunking | Retrieval | MRR | Hit@1 | Hit@ctx | ΔMRR vs best (paired) |
|---|---|---|---|---|---|
| fixed | hybrid_rerank | 0.84 [0.75, 0.92] | 0.76 | 0.96 | reference |
| structure | hybrid_rerank | 0.83 [0.73, 0.91] | 0.73 | 0.98 | -0.015 [-0.094, +0.059] |
| semantic | hybrid | 0.80 [0.70, 0.89] | 0.73 | 0.98 | -0.040 [-0.137, +0.045] |
| structure | hybrid | 0.79 [0.69, 0.87] | 0.67 | 0.96 | -0.056 [-0.160, +0.050] |
| fixed | hybrid | 0.77 [0.67, 0.87] | 0.67 | 0.96 | -0.067 [-0.172, +0.036] |

Top 5 of 12 configurations; full tables in [`results/retrieval_metrics.md`](results/retrieval_metrics.md).
<!-- END:retrieval-top -->

### Generation and RAGAS

<!-- BEGIN:generation -->
_Pending: all runs judged by RAGAS, then `python -m ragk8s compare`._
<!-- END:generation -->

### Application generator vs reference generator

<!-- BEGIN:models -->
_Pending: reference generator run fully judged, then `python -m ragk8s compare`._
<!-- END:models -->

### How far can the judge be trusted?

<!-- BEGIN:agreement -->
_Pending: human labels (`python -m ragk8s validation export`, then `agree`)._
<!-- END:agreement -->

Methodology, run conditions and limitations: [REPORT.md](REPORT.md).

## Pipeline

```mermaid
flowchart LR
    A["Kubernetes docs<br/>frozen snapshot, 408 pages"] --> B["Markup cleaning"]
    B --> C1["Fixed chunks"]
    B --> C2["Structure chunks"]
    B --> C3["Semantic chunks"]
    C1 & C2 & C3 --> D["BM25 + FAISS indexes"]
    D --> E["Retrieve: BM25 / dense / RRF / RRF + rerank<br/>same 5,000-char budget"]
    E --> F["Generate with [n] citations<br/>or the refusal sentence"]
    F --> G["RAGAS judge"]
    E --> H["Retrieval metrics, no LLM"]
    G & H --> I["Bootstrap CIs, paired comparisons"]
```

| Stage | Choice |
|---|---|
| Corpus | Kubernetes docs (concepts, tasks, tutorials) at upstream commit `5184b9b`, 408 pages, reproducible byte for byte |
| Chunking | fixed 1,000 chars (150 overlap) · Markdown `##`/`###` sections (≤ 1,500 chars) · semantic (sentence-embedding similarity ≥ 0.55) |
| Retrieval | `rank_bm25` · `all-MiniLM-L6-v2` + FAISS · Reciprocal Rank Fusion (k = 60) · `ms-marco-MiniLM-L-6-v2` cross-encoder |
| Context | ranked chunks up to 5,000 characters for every strategy (fair across chunk sizes) |
| Generators | `qwen/qwen3.8-27b` on Groq (reasoning off) · `qwen2.5:7b-instruct` with Ollama as reference |
| Judge | `gemma4:12b-it-q8_0` with Ollama (another model family), RAGAS 0.4.3 |
| Gold set | 45 factual questions (18 drafted from whole pages, 27 from chunks) + 8 out-of-corpus questions |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m ragk8s ingest --verify       # check the frozen corpus
python -m ragk8s chunk && python -m ragk8s index
python -m ragk8s retrieve --all && python -m ragk8s retrieval-metrics
python -m ragk8s generate --provider groq --preset main   # needs GROQ_API_KEY; resumable, quota-aware
python -m ragk8s compare
```

Every step, including the Kaggle notebooks for the reference generator and the
judge, is documented in [docs/REPRODUCE.md](docs/REPRODUCE.md). Runs are cached and
resumable; each run records model digest, decoding options, code commit and timing.

## Repository layout

```
src/ragk8s/        package: ingestion, chunking, indexing, retrieval, generation,
                   pipeline, judge, statistics, comparison, CLI
tests/             unit tests (no model, no network)
eval/              gold questions, review sheets, dropped questions with reasons
results/           metrics, comparison tables, figures, summary.json
kaggle/            notebooks for the GPU work (reference generator, judge)
data/raw/          frozen corpus snapshot (CC BY 4.0, see NOTICE)
```

## Limitations (short version)

The gold set is small (45 factual questions) and its review was LLM-assisted, then
validated by the author; the application generator is a preview model on a free
API tier; RAGAS scores come from a single 12B judge whose agreement with a human is
reported above. Details in [REPORT.md](REPORT.md#limitations).

## License and attribution

- Documentation corpus: © The Kubernetes Authors, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
  modified (front matter removed, files renamed); see [NOTICE](NOTICE).
- Unofficial project, not affiliated with the Linux Foundation or the Kubernetes project.
- Code license: not chosen yet.
