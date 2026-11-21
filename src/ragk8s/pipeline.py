"""Run retrieval + generation over the evaluation questions for one configuration.

Output: results/run_<strategy>_<method>.jsonl with, per question, the answer,
the retrieved contexts, the cited sources, the ground truth and the category
(factual / impossible).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import config, generation
from .config import Method, Strategy
from .retrieval import Retriever
from .utils import read_jsonl, write_jsonl


def load_questions(
    path: Path = config.EVAL_DIR / "questions.jsonl", limit_factual: int | None = None
) -> list[dict[str, Any]]:
    questions = read_jsonl(path)
    factual = [q for q in questions if q.get("category", "factual") == "factual"]
    impossible = [q for q in questions if q.get("category") == "impossible"]
    if limit_factual is not None:
        factual = factual[:limit_factual]
    return factual + impossible


def run_name(strategy: Strategy, method: Method) -> str:
    return f"run_{strategy}_{method}"


def run(strategy: Strategy, method: Method, top_k: int = config.TOP_K, limit_factual: int | None = None) -> Path:
    from tqdm import tqdm

    questions = load_questions(limit_factual=limit_factual)
    retriever = Retriever(strategy)
    records = []
    for q in tqdm(questions, desc=run_name(strategy, method)):
        hits = retriever.search(method, q["question"], top_k=top_k)
        records.append(
            {
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "category": q.get("category", "factual"),
                "answer": generation.generate(q["question"], hits),
                "contexts": [h.chunk["text"] for h in hits],
                "sources": generation.sources(hits),
            }
        )
    out = config.RESULTS_DIR / f"{run_name(strategy, method)}.jsonl"
    write_jsonl(out, records)
    print(f"{len(records)} answers -> {out}")
    return out
