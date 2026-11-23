"""Build the data bundle of the DocTruth web app from the evaluation results.

The app never recomputes what the evaluation measured. This module copies, into
one JSON file (app_data/bundle.json), only what the app shows:

- the evaluation questions (examples), with the page each was written from;
- for every evaluation question, the answers of the evaluated models with the
  best configuration, and their sources (precomputed answers);
- for every evaluation question and strategy, the hybrid + rerank context of the
  evaluation run (the reranker is too slow to run live on one vCPU, see
  results/onnx_parity_*.json);
- the measured scores: retrieval metrics, results/summary.json once the judge run
  is complete, the ONNX parity check.

A missing input is left out (null), never replaced by an estimate.

    python -m ragk8s app-bundle   -> app_data/bundle.json
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import config
from .pipeline import BEST_CONFIG, RETRIEVAL_DIR, RUNS_DIR, RunConfig, _git_commit, load_questions
from .utils import read_jsonl, write_json

BUNDLE_PATH = Path("app_data") / "bundle.json"
LIVE_CONFIG = RunConfig("fixed", "hybrid")
MODELS = (config.GROQ_MODEL.split("/")[-1], config.OLLAMA_MODEL.replace(":", "-"))


def _source(n: int, hit: dict[str, Any]) -> dict[str, Any]:
    score = hit.get("rerank_score")
    return {
        "n": n,
        "title": hit["title"],
        "url": hit["source_url"],
        "doc_id": hit["doc_id"],
        "chunk_id": hit["chunk_id"],
        "text": hit["text"],
        "score": float(score if score is not None else hit["score"]),
    }


def _retrieval(cfg: RunConfig, root: Path) -> dict[str, list[dict[str, Any]]]:
    path = root / f"{cfg.name}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: [_source(i, h) for i, h in enumerate(r["hits"], 1)] for r in read_jsonl(path)}


def _answers(model: str, cfg: RunConfig, root: Path) -> tuple[dict[str, dict[str, Any]], str | None]:
    path = root / model / f"run_{cfg.name}.jsonl"
    meta_path = path.with_suffix(".meta.json")
    if not path.exists() or not meta_path.exists():
        return {}, None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("complete"):
        return {}, None
    answers = {
        r["id"]: {
            "model": model,
            "text": r["answer"],
            "is_refusal": bool(r["is_refusal"]),
            "cited": [int(n) for n in r["cited"]],
            "origin": "precomputed",
            "generated_at": meta.get("created_at"),
        }
        for r in read_jsonl(path)
    }
    return answers, meta.get("created_at")


def _estimate(row: dict[str, str], name: str) -> dict[str, Any] | None:
    if not row.get(name):
        return None
    return {
        "mean": float(row[name]),
        "low": float(row[f"{name}_low"]),
        "high": float(row[f"{name}_high"]),
        "n": int(row["n"]),
    }


def _retrieval_rows(path: Path) -> list[dict[str, Any]]:
    """results/retrieval_metrics.csv: one row per (subset, config); subset "all" is the main table."""
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            diff = None
            if row.get("mrr_diff_vs_ref"):
                diff = {
                    "mean": float(row["mrr_diff_vs_ref"]),
                    "low": float(row["mrr_diff_low"]),
                    "high": float(row["mrr_diff_high"]),
                    "n": int(row["n"]),
                }
            rows.append(
                {
                    "subset": row.get("subset") or "all",
                    "config": row["config"],
                    "chunking": row["chunking"],
                    "retrieval": row["retrieval"],
                    "n": int(row["n"]),
                    "mrr": _estimate(row, "mrr"),
                    "hit_at_1": _estimate(row, "hit@1"),
                    "hit_at_3": _estimate(row, "hit@3"),
                    "hit_at_ctx": _estimate(row, "hit@ctx"),
                    "context_chars": float(row["context_chars"]),
                    "mrr_diff_vs_best": diff,
                }
            )
    return rows


JUDGE_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _generation(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    """results/summary.json, with the judge metrics of a config left null until its judge run is complete."""
    if not summary:
        return None
    rows = [r for r in [*summary.get("configs", []), summary.get("reference")] if r]
    for row in rows:
        if not row.get("judge_complete"):
            for metric in JUDGE_METRICS:
                row[metric] = None
    complete = bool(rows) and all(r.get("judge_complete") for r in rows)
    return {
        "app_model": summary.get("app_model"),
        "reference_model": summary.get("reference_model"),
        "best_config": summary.get("best_config"),
        "judge": summary.get("judge"),
        "judge_complete": complete,
        "configs": summary.get("configs", []),
        "reference": summary.get("reference"),
        "judge_agreement": summary.get("judge_agreement") if complete else None,
        "generated_at": summary.get("generated_at"),
    }


def _json_or_none(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build(
    out: Path = BUNDLE_PATH,
    results: Path = config.RESULTS_DIR,
    retrieval_dir: Path = RETRIEVAL_DIR,
    runs_dir: Path = RUNS_DIR,
) -> dict[str, Any]:
    questions = load_questions()
    best_sources = _retrieval(BEST_CONFIG, retrieval_dir)
    rerank = {s: _retrieval(RunConfig(s, "hybrid_rerank"), retrieval_dir) for s in config.STRATEGIES}
    per_model = {m: _answers(m, BEST_CONFIG, runs_dir) for m in MODELS}

    precomputed: dict[str, Any] = {}
    for q in questions:
        answers = [per_model[m][0][q["id"]] for m in MODELS if q["id"] in per_model[m][0]]
        if q["id"] in best_sources and answers:
            precomputed[q["id"]] = {"config": BEST_CONFIG.name, "sources": best_sources[q["id"]], "answers": answers}

    generation = _generation(_json_or_none(results / "summary.json"))
    bundle: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "live_config": LIVE_CONFIG.name,
        "best_config": BEST_CONFIG.name,
        "budget_chars": config.CONTEXT_BUDGET_CHARS,
        "questions": [
            {
                "id": q["id"],
                "question": q["question"],
                "category": q["category"],
                "source_doc_id": q.get("source_doc_id"),
                "source_url": q.get("source_url"),
            }
            for q in questions
        ],
        "precomputed": precomputed,
        "rerank": rerank,
        "scores": {
            "retrieval": _retrieval_rows(results / "retrieval_metrics.csv"),
            # Judge metrics stay null until the judge run is complete: no partial scores in the app.
            "generation": generation,
            "onnx_parity": _json_or_none(results / "onnx_parity.json"),
        },
    }
    write_json(out, bundle)
    return {
        "path": str(out),
        "questions": len(questions),
        "precomputed_answers": len(precomputed),
        "rerank_contexts": {s: len(v) for s, v in rerank.items()},
        "models_with_answers": [m for m in MODELS if per_model[m][0]],
        "retrieval_rows": len(bundle["scores"]["retrieval"]),
        "generation_summary": generation is not None,
        "judge_complete": bool(generation and generation["judge_complete"]),
    }
