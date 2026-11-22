"""RAGAS evaluation of generation runs with a local judge model (run on a Kaggle GPU).

Metrics (RAGAS 0.4 "collections" API):
- faithfulness       answer claims supported by the retrieved contexts
- answer_relevancy   answer addresses the question (judge LLM + embeddings)
- context_precision  relevant contexts ranked first (needs the reference answer)
- context_recall     reference answer covered by the contexts

Scope: factual questions only. faithfulness and answer_relevancy are computed on
answers that are not refusals (a wrongly refused factual question is counted by
the false refusal rate instead); context metrics depend only on retrieval and are
computed for every factual question. Refusals on out-of-corpus questions are
measured outside RAGAS (refusal.py).

Every (metric, inputs, judge) score is cached in .cache/judge/<key>.json, so a
Kaggle session can stop at any time (``max_minutes``) and the next one resumes.
Context metrics do not depend on the generator, so the 7B and 27B runs of the same
configuration share them.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__, config
from .utils import read_jsonl, sha256_text, write_json, write_jsonl

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
JUDGE_MODEL = "gemma4:12b-it-q8_0"
JUDGE_EMBEDDINGS = config.EMBEDDING_MODEL
JUDGE_CACHE = config.CACHE_DIR / "judge"
SCORES_DIR = config.RESULTS_DIR / "judge"
OLLAMA_V1 = "http://127.0.0.1:11434/v1"
# Sent to Ollama's OpenAI-compatible endpoint: top-level reasoning_effort="none" disables
# thinking there (a think=false field is ignored on /v1).
JUDGE_LLM_KWARGS: dict[str, Any] = {"temperature": 0.0, "max_tokens": 2048, "reasoning_effort": "none"}

ScoreFn = Callable[[str, Mapping[str, Any]], Awaitable[float | None]]


@dataclass(frozen=True)
class JudgeId:
    """Everything that defines a judge; part of every cache key."""

    model: str
    digest: str | None
    ragas_version: str
    embeddings: str

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.model, "digest": self.digest, "ragas": self.ragas_version, "embeddings": self.embeddings}


def metric_inputs(record: Mapping[str, Any], metric: str) -> dict[str, Any] | None:
    """RAGAS inputs for one run record, or None when the metric does not apply."""
    if record.get("category") != "factual":
        return None
    question, contexts, reference = record["question"], list(record["contexts"]), record["ground_truth"]
    if metric in ("faithfulness", "answer_relevancy"):
        if record.get("is_refusal"):
            return None
        if metric == "faithfulness":
            return {"user_input": question, "response": record["answer"], "retrieved_contexts": contexts}
        return {"user_input": question, "response": record["answer"]}
    if metric == "context_precision":
        return {"user_input": question, "reference": reference, "retrieved_contexts": contexts}
    if metric == "context_recall":
        return {"user_input": question, "retrieved_contexts": contexts, "reference": reference}
    raise ValueError(f"unknown metric: {metric}")


def cache_key(metric: str, inputs: Mapping[str, Any], judge: JudgeId) -> str:
    return sha256_text(json.dumps({"metric": metric, "inputs": inputs, "judge": judge.as_dict()}, sort_keys=True))


@dataclass
class Task:
    run: str
    record_id: str
    metric: str
    inputs: dict[str, Any]
    key: str


def plan(run_files: Sequence[Path], judge: JudgeId, limit: int | None = None) -> list[Task]:
    tasks: list[Task] = []
    for path in run_files:
        records = read_jsonl(path)[:limit] if limit else read_jsonl(path)
        for record in records:
            for metric in METRICS:
                inputs = metric_inputs(record, metric)
                if inputs is not None:
                    tasks.append(Task(str(path), record["id"], metric, inputs, cache_key(metric, inputs, judge)))
    return tasks


def _cached(cache_dir: Path, key: str) -> dict[str, Any] | None:
    path = cache_dir / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


async def _run_tasks(
    tasks: Sequence[Task],
    score: ScoreFn,
    cache_dir: Path,
    concurrency: int,
    deadline: float | None,
    retry_errors: bool,
    progress: Callable[[], None] | None = None,
) -> dict[str, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats = {"cached": 0, "scored": 0, "errors": 0, "skipped_deadline": 0}
    semaphore = asyncio.Semaphore(concurrency)
    unique: dict[str, Task] = {}
    for t in tasks:
        unique.setdefault(t.key, t)

    async def one(task: Task) -> None:
        entry = _cached(cache_dir, task.key)
        if entry is not None and (entry.get("error") is None or not retry_errors):
            stats["cached"] += 1
            if progress:
                progress()
            return
        async with semaphore:
            if deadline is not None and time.monotonic() > deadline:
                stats["skipped_deadline"] += 1
                return
            started = time.monotonic()
            try:
                value = await asyncio.wait_for(score(task.metric, task.inputs), timeout=900)
                entry = {"metric": task.metric, "value": value, "error": None}
                stats["scored"] += 1
            except Exception as exc:  # judge parsing failures are data, not crashes
                entry = {"metric": task.metric, "value": None, "error": f"{type(exc).__name__}: {exc}"[:500]}
                stats["errors"] += 1
            entry["seconds"] = round(time.monotonic() - started, 2)
            write_json(cache_dir / f"{task.key}.json", entry)
            if progress:
                progress()

    await asyncio.gather(*(one(t) for t in unique.values()))
    return stats


def assemble(
    run_files: Sequence[Path], judge: JudgeId, cache_dir: Path, out_dir: Path, meta: dict[str, Any]
) -> list[Path]:
    """Write one scores file per run from the cache (missing scores stay null)."""
    outputs = []
    for path in run_files:
        rows = []
        for record in read_jsonl(path):
            if record.get("category") != "factual":
                continue
            row: dict[str, Any] = {k: record.get(k) for k in ("id", "question", "origin", "is_refusal")}
            for metric in METRICS:
                inputs = metric_inputs(record, metric)
                entry = _cached(cache_dir, cache_key(metric, inputs, judge)) if inputs is not None else None
                row[metric] = entry.get("value") if entry else None
                row[f"{metric}_status"] = (
                    "n/a" if inputs is None else "missing" if entry is None else "error" if entry.get("error") else "ok"
                )
            rows.append(row)
        model_dir, run_name = path.parent.name, path.stem.removeprefix("run_")
        out = out_dir / model_dir / f"scores_{run_name}.jsonl"
        write_jsonl(out, rows)
        counts = {
            m: {s: sum(1 for r in rows if r[f"{m}_status"] == s) for s in ("ok", "error", "missing", "n/a")}
            for m in METRICS
        }
        write_json(out.with_suffix(".meta.json"), {**meta, "run": str(path), "n_factual": len(rows), "status": counts})
        outputs.append(out)
    return outputs


# --- Real judge (RAGAS + Ollama), imported only where it runs -----------------------


def ollama_digest(model: str) -> str | None:
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as resp:
            for entry in json.load(resp).get("models", []):
                if entry.get("name") == model or entry.get("model") == model:
                    return str(entry.get("digest"))
    except OSError:
        return None
    return None


def ragas_scorer(model: str = JUDGE_MODEL, embeddings_model: str = JUDGE_EMBEDDINGS) -> tuple[ScoreFn, JudgeId]:
    import ragas
    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness

    client = AsyncOpenAI(base_url=OLLAMA_V1, api_key="ollama", timeout=600.0, max_retries=1)
    llm = llm_factory(model, provider="openai", client=client, **JUDGE_LLM_KWARGS)
    embeddings = HuggingFaceEmbeddings(model=embeddings_model)
    metrics: dict[str, Any] = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    async def score(metric: str, inputs: Mapping[str, Any]) -> float | None:
        result = await metrics[metric].ascore(**inputs)
        value = result.value
        return None if value is None else float(value)

    judge = JudgeId(
        model=model, digest=ollama_digest(model), ragas_version=ragas.__version__, embeddings=embeddings_model
    )
    return score, judge


def run(
    run_files: Sequence[Path],
    max_minutes: float | None = None,
    concurrency: int = 4,
    limit: int | None = None,
    retry_errors: bool = False,
    cache_dir: Path = JUDGE_CACHE,
    out_dir: Path = SCORES_DIR,
    scorer: tuple[ScoreFn, JudgeId] | None = None,
) -> dict[str, Any]:
    from tqdm import tqdm

    score, judge = scorer or ragas_scorer()
    tasks = plan(run_files, judge, limit)
    deadline = time.monotonic() + 60 * max_minutes if max_minutes else None
    started = time.monotonic()
    bar = tqdm(total=len({t.key for t in tasks}), desc="judge")
    stats = asyncio.run(
        _run_tasks(tasks, score, cache_dir, concurrency, deadline, retry_errors, progress=lambda: bar.update(1))
    )
    bar.close()
    elapsed = time.monotonic() - started
    meta = {
        "judge": judge.as_dict(),
        "llm_kwargs": JUDGE_LLM_KWARGS,
        "metrics": list(METRICS),
        "concurrency": concurrency,
        "ragk8s_version": __version__,
        "git_commit": os.environ.get("RAGK8S_GIT_COMMIT"),
        "last_session": {**stats, "seconds": round(elapsed, 1)},
        "seconds_per_new_score": round(elapsed / stats["scored"], 2) if stats["scored"] else None,
    }
    outputs = assemble(run_files, judge, cache_dir, out_dir, meta)
    remaining = len({t.key for t in tasks}) - stats["cached"] - stats["scored"] - stats["errors"]
    return {
        **stats,
        "unique_tasks": len({t.key for t in tasks}),
        "remaining": remaining,
        "seconds": round(elapsed, 1),
        "outputs": [str(o) for o in outputs],
    }
