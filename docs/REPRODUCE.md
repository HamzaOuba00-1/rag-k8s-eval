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

Both are deterministic for a given environment (seed in `src/ragk8s/config.py`).
`fixed` and `structure` chunking do not use any model; `semantic` chunking and the
dense index use `sentence-transformers/all-MiniLM-L6-v2`.

## 4. Retrieval and generation

```bash
python -m ragk8s search "How do I create a Deployment?" --strategy structure
python -m ragk8s ask "How do I create a Deployment?" --strategy structure --method hybrid_rerank
python -m ragk8s run --strategy structure --method hybrid_rerank
```

## 5. Evaluation

Being reworked (resumable runs, retrieval metrics, RAGAS with caching, judge
validation, bootstrap confidence intervals). The previous scripts are kept in
`legacy/` until they are replaced.

## Quality checks

```bash
ruff check src tests
ruff format --check src tests
mypy
pytest
```
