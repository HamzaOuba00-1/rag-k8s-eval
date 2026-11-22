from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ragk8s import judge
from ragk8s.judge import JudgeId, metric_inputs, plan

JUDGE = JudgeId("gemma", "d", "0.4.3", "emb")


def _record(rid: str, category: str = "factual", refusal: bool = False) -> dict[str, Any]:
    return {
        "id": rid,
        "question": f"{rid}?",
        "category": category,
        "origin": "page",
        "is_refusal": refusal,
        "answer": "A [1].",
        "contexts": ["c1", "c2"],
        "ground_truth": "A.",
    }


def test_metric_inputs_scope() -> None:
    assert metric_inputs(_record("q", category="impossible"), "faithfulness") is None
    refused = _record("q", refusal=True)
    assert metric_inputs(refused, "faithfulness") is None and metric_inputs(refused, "answer_relevancy") is None
    assert metric_inputs(refused, "context_recall") == {
        "user_input": "q?",
        "retrieved_contexts": ["c1", "c2"],
        "reference": "A.",
    }
    assert set(metric_inputs(_record("q"), "faithfulness") or {}) == {"user_input", "response", "retrieved_contexts"}


def _write_run(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_context_scores_are_shared_between_generators(tmp_path: Path) -> None:
    a = _write_run(tmp_path / "runs" / "m27" / "run_fixed_hybrid_rerank.jsonl", [_record("q1")])
    b = _write_run(tmp_path / "runs" / "m7" / "run_fixed_hybrid_rerank.jsonl", [{**_record("q1"), "answer": "B [2]."}])
    tasks = plan([a, b], JUDGE)
    assert len(tasks) == 8 and len({t.key for t in tasks}) == 6  # 2 context metrics shared


def test_run_caches_resumes_and_records_errors(tmp_path: Path) -> None:
    run_file = _write_run(
        tmp_path / "runs" / "m" / "run_fixed_bm25.jsonl",
        [_record("q1"), _record("q2", refusal=True), _record("q3", category="impossible")],
    )
    calls: list[str] = []

    async def score(metric: str, inputs: Mapping[str, Any]) -> float | None:
        calls.append(metric)
        if metric == "answer_relevancy":
            raise ValueError("unparsable judge output")
        return 0.5

    kwargs = {"cache_dir": tmp_path / "cache", "out_dir": tmp_path / "scores", "scorer": (score, JUDGE)}
    stats = judge.run([run_file], **kwargs)  # type: ignore[arg-type]
    assert stats["unique_tasks"] == 6 and stats["errors"] == 1 and stats["scored"] == 5 and stats["remaining"] == 0

    rows = [
        json.loads(line) for line in (tmp_path / "scores" / "m" / "scores_fixed_bm25.jsonl").read_text().splitlines()
    ]
    assert [r["id"] for r in rows] == ["q1", "q2"]  # out-of-corpus questions are not scored
    assert rows[0]["faithfulness"] == 0.5 and rows[0]["answer_relevancy_status"] == "error"
    assert rows[1]["faithfulness_status"] == "n/a" and rows[1]["context_recall"] == 0.5

    calls.clear()
    judge.run([run_file], **kwargs)  # type: ignore[arg-type]
    assert calls == []  # everything cached, errors included
    judge.run([run_file], retry_errors=True, **kwargs)  # type: ignore[arg-type]
    assert calls == ["answer_relevancy"]


def test_deadline_stops_new_scores(tmp_path: Path) -> None:
    run_file = _write_run(tmp_path / "runs" / "m" / "run_fixed_bm25.jsonl", [_record("q1")])

    async def score(metric: str, inputs: Mapping[str, Any]) -> float | None:
        return 1.0

    stats = judge.run(
        [run_file], max_minutes=-1, cache_dir=tmp_path / "c", out_dir=tmp_path / "s", scorer=(score, JUDGE)
    )
    assert stats["scored"] == 0 and stats["skipped_deadline"] == 4 and stats["remaining"] == 4
