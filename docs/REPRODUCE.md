# Reproducing the pipeline

All commands run from the repository root. Python 3.11+ (developed with 3.12).

## 1. Environment

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1   |   Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

Generation needs [Ollama](https://ollama.com/download) and the model:

```bash
ollama pull qwen2.5:7b-instruct
```

## 2. Corpus

`data/raw/` is a frozen snapshot of the Kubernetes documentation (concepts, tasks,
tutorials; 408 documents) taken at upstream commit
`5184b9b24b288709ed36b4003f53dc1530eda2f3` (recorded in `data/raw/corpus.json`).

```bash
python -m ragk8s ingest --verify   # offline: check every file against its SHA-256
python -m ragk8s ingest            # re-download at the pinned commit and compare with the snapshot
```

`ingest` never overwrites the snapshot unless `--force` is given.

## 3. Chunking and indexing

```bash
python -m ragk8s chunk    # data/processed/chunks_{fixed,structure,semantic}.jsonl
python -m ragk8s index    # data/index/<strategy>/ (BM25 tokens, FAISS, metadata)
```

Before chunking, Hugo shortcodes and layout comments are rendered to plain text
(`src/ragk8s/textclean.py`); `data/raw/` itself is never modified.

Both are deterministic for a given environment (seed in `src/ragk8s/config.py`).
`fixed` and `structure` chunking do not use any model; `semantic` chunking and the
dense index use `sentence-transformers/all-MiniLM-L6-v2`.

## 4. Retrieval and generation

Interactive checks:

```bash
python -m ragk8s search "How do I create a Deployment?" --strategy structure
python -m ragk8s ask "How do I create a Deployment?" --strategy structure --method hybrid_rerank
```

Decoding is deterministic (temperature 0, seed 42, `num_ctx` 4096). `ask` also prints
whether the answer is a refusal, which sources it cites, the share of sentences
with a citation, token counts and the generation time.

## 5. Evaluation runs (two phases)

```bash
# Phase 1, no LLM: context of every evaluation question, for all 12 configurations
# (ranked chunks cut to the same 5,000-character budget for every strategy)
python -m ragk8s retrieve --all

# Phase 2, Ollama: answers from the stored contexts
python -m ragk8s generate --configs structure:hybrid_rerank
```

- Retrieval results go to `results/retrieval/<strategy>_<method>.jsonl` (+ `.meta.json`).
- Each answer is cached in `.cache/generations/`, keyed by model digest, prompt,
  decoding options, question and contexts. `generate` can be interrupted (Ctrl+C,
  crash, reboot) and rerun: finished answers are reused.
- Runs are written to `results/runs/run_<strategy>_<method>.jsonl` with a `.meta.json`
  (model digest, Ollama version, options, timings, git commit, platform).
- The phases are separate so that the embedding and reranking models are not in
  memory while the LLM runs (`run` chains both).

Generators:

```bash
# Application generator: qwen/qwen3.8-27b on Groq (reasoning disabled, temperature 0)
python -m ragk8s generate --provider groq --preset main     # the 6 evaluated configurations
# Reference generator: qwen2.5:7b-instruct with Ollama, on the best configuration only
python -m ragk8s generate --provider ollama --preset best
```

The reference 7B run happens on a Kaggle GPU (the local GPU is too slow):

```bash
python -m ragk8s kaggle export --preset best   # kaggle/export/contexts-fixed_hybrid_rerank.zip
# upload the zip as a Kaggle dataset, attach it to kaggle/generate_ollama.ipynb, Run All,
# download ragk8s-output.zip from the Output panel, then:
python -m ragk8s kaggle import path/to/ragk8s-output.zip
```

The export records the git commit; the notebook installs `ragk8s` at that commit, so
the prompt, decoding options and cache keys are the same as for local runs.

Runs are written per model: `results/runs/<model>/run_<strategy>_<method>.jsonl`.
With the Groq free tier, `generate` stops cleanly when the daily token budget of the
model is reached ("Daily quota reached"); rerun the same command the next day and it
continues from the cache.

## 6. RAGAS judge (Kaggle GPU)

Judge: `gemma4:12b-it-q8_0` with Ollama, a different model family from both generators,
thinking disabled (`reasoning_effort="none"`), temperature 0. RAGAS 0.4.3 is pinned
(`pip install ".[judge]"`; it needs langchain-community 0.4.1). Metrics: faithfulness and
answer_relevancy on factual answers that are not refusals; context_precision and
context_recall on every factual question; refusal rates are computed outside RAGAS.

```bash
python -m ragk8s kaggle export --preset main --runs qwen3.8-27b qwen2.5-7b-instruct --with-judge-cache --name judge-input
# Kaggle: attach the dataset to kaggle/judge_ragas.ipynb, MODE="pilot" then "full"
python -m ragk8s kaggle import path/to/ragk8s-judge-output.zip   # results/judge/<model>/scores_<config>.jsonl
```

Only complete runs are exported. Each score is cached by metric, inputs and judge
(model digest, RAGAS version, embeddings); context metrics are shared between the two
generators. Re-export with `--with-judge-cache` to resume in a new session.

### Judge validation (human labels)

```bash
python -m ragk8s validation export --n 40   # eval/review/judge_validation.xlsx (judge scores hidden)
python -m ragk8s validation agree           # results/judge_agreement.json
```

A seeded sample stratified by generator and by judge faithfulness (< 1 vs = 1) is
labelled blind on three levels (0 / 0.5 / 1): claims supported by the contexts,
answer relevant to the question, contexts covering the reference answer. Agreement:
Spearman correlation (bootstrap CI), mean absolute error, judge-minus-human bias and
Cohen's kappa after binarising at 0.5.

### Final comparison

```bash
python -m ragk8s compare   # results/comparison.{md,csv}, results/summary.json, results/figures/*.png
```

Every number is read from the run, judge and agreement files; configurations whose
runs are incomplete appear as missing rather than estimated.

```bash
python -m ragk8s report    # regenerates the result blocks of README.md and REPORT.md
```

The documentation never contains hand-typed results: each block between
`<!-- BEGIN:name -->` and `<!-- END:name -->` is rebuilt from the result files, and stays
"pending" until its source is complete (e.g. every configuration fully judged).

## 7. Extra questions drafted from whole pages (Groq)

The first 55 factual questions were drafted from single `structure` chunks. To get a
subset that no chunking strategy has seen, extra questions are drafted from whole,
unused pages (seeded sample, proportional to concepts / tasks / tutorials) by
`openai/gpt-oss-120b` on Groq, a different model family from the evaluated generator.

```bash
python -m ragk8s questions draft --n 25
```

The Groq API key is read from the `GROQ_API_KEY` environment variable. When it is not
set, the command asks for it with hidden input (it is kept in memory for this process
only). Never put the key in a file of the repository.

Drafts whose evidence sentence is not found verbatim in the page, or whose question
refers to "this page", are rejected automatically (`eval/review/drafts_rejected.jsonl`);
the others are appended to `eval/questions.jsonl` with `origin: page` and
`reviewed: false`. The Groq client paces requests under the per-minute token limit,
keeps a per-day token ledger in `.cache/groq_usage.json`, and stops cleanly when the
daily quota is reached; rerunning resumes from the cache.

## 8. Gold questions review

```bash
python -m ragk8s review export   # eval/review/questions_review.xlsx (keep / fix / drop per question)
python -m ragk8s review apply    # writes the verdicts back to eval/questions.jsonl
```

`apply` refuses to write anything while a verdict is missing. Dropped questions are
kept in `eval/questions_dropped.jsonl`. After a review, rerun `retrieve --all`; only
the edited questions are regenerated by `generate`.

## 9. Retrieval metrics (no LLM)

```bash
python -m ragk8s retrieval-metrics            # results/retrieval_metrics.{md,csv}
```

Hit@1, Hit@3, Hit@ctx (source page anywhere in the context) and MRR of the question's
source page, with 95% bootstrap intervals and paired MRR differences against the best
configuration; reported on all factual questions and separately per question origin.

The RAGAS evaluation is being reworked; the previous scripts are kept in `legacy/`.

## 10. ONNX models for the web app

The web app runs retrieval without PyTorch: the embedding model and the reranker are
exported once to ONNX, in three variants, then checked against PyTorch.

| Variant | File | Quantisation |
|---|---|---|
| `fp32` | `model.onnx` | none |
| `int8` | `model.int8.onnx` | dynamic int8 on every supported op, word embeddings included |
| `int8-matmul` | `model.int8-matmul.onnx` | dynamic int8, per channel, MatMul weights only |

```bash
pip install -e ".[dev,onnx]"
python -m ragk8s onnx export   # models/onnx/{embedder,reranker}/ (three variants, tokenizer.json)
python -m ragk8s onnx check    # results/onnx_parity.json, 1 CPU thread
python -m ragk8s onnx check --strategies fixed --variants fp32 int8-matmul --threads 2
python -m ragk8s onnx check --strategies fixed --variants int8-matmul --rerank-max-length 256
```

`check` reports, for every strategy, model-based method and variant, the share of
evaluation questions that get exactly the same context as with PyTorch, the mean
Jaccard overlap, the source-page MRR and the time per query. PyTorch and onnxruntime
run with the same number of threads (`--threads`, default 1 as on a one-vCPU Cloud Run
instance), so the times can be compared. `--rerank-max-length` shortens only the ONNX
reranker input; PyTorch keeps the evaluation settings, so the agreement figures show how
far a faster setting drifts from what was evaluated.

## Quality checks

CI (`.github/workflows/ci.yml`) runs the same checks on every push, with a light
install: no PyTorch, no model download, no LLM.

```bash
ruff check src tests
ruff format --check src tests
mypy
pytest
```
