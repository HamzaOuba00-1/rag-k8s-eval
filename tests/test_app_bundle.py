from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ragk8s import app_bundle
from ragk8s.utils import write_json, write_jsonl

QUESTIONS = [
    {"id": "q1", "question": "Q one?", "category": "factual", "source_doc_id": "d1", "source_url": "https://k/1"},
    {"id": "q2", "question": "Q two?", "category": "impossible"},
]


def _hit(rank: int, rerank: float | None) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": 0.03,
        "rerank_score": rerank,
        "chunk_id": f"c{rank}",
        "doc_id": "d1",
        "title": "T",
        "source_url": "https://k/1",
        "text": "x",
    }


def _run(path: Path, complete: bool) -> None:
    write_jsonl(path, [{"id": "q1", "answer": "A [1].", "is_refusal": False, "cited": [1]}])
    write_json(path.with_suffix(".meta.json"), {"complete": complete, "created_at": "2026-11-24T09:00:00+00:00"})


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(app_bundle, "load_questions", lambda: QUESTIONS)
    retrieval = tmp_path / "results" / "retrieval"
    for strategy in ("fixed", "structure", "semantic"):
        write_jsonl(retrieval / f"{strategy}_hybrid_rerank.jsonl", [{"id": "q1", "hits": [_hit(1, 6.5)]}])
    runs = tmp_path / "results" / "runs"
    _run(runs / app_bundle.MODELS[0] / "run_fixed_hybrid_rerank.jsonl", complete=True)
    _run(runs / app_bundle.MODELS[1] / "run_fixed_hybrid_rerank.jsonl", complete=False)
    return tmp_path


def test_bundle_keeps_only_complete_runs_and_measured_scores(tree: Path) -> None:
    results = tree / "results"
    out = tree / "app_data" / "bundle.json"
    report = app_bundle.build(out, results, results / "retrieval", results / "runs")
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert report["models_with_answers"] == [app_bundle.MODELS[0]]  # incomplete 7B run left out
    pre = bundle["precomputed"]["q1"]
    assert [a["model"] for a in pre["answers"]] == [app_bundle.MODELS[0]]
    assert pre["sources"][0] == {
        "n": 1,
        "title": "T",
        "url": "https://k/1",
        "doc_id": "d1",
        "chunk_id": "c1",
        "text": "x",
        "score": 6.5,
    }
    assert "q2" not in bundle["precomputed"]
    assert bundle["scores"] == {"retrieval": [], "generation": None, "onnx_parity": None}
    assert bundle["live_config"] == "fixed_hybrid" and bundle["best_config"] == "fixed_hybrid_rerank"


def test_generation_scores_are_left_out_until_the_judge_run_is_complete(tree: Path) -> None:
    results = tree / "results"
    write_json(results / "summary.json", {"configs": [{"judge_complete": True}, {"judge_complete": False}]})
    out = tree / "bundle.json"
    assert app_bundle.build(out, results, results / "retrieval", results / "runs")["generation_scores"] is False
    write_json(results / "summary.json", {"configs": [{"judge_complete": True}]})
    assert app_bundle.build(out, results, results / "retrieval", results / "runs")["generation_scores"] is True
