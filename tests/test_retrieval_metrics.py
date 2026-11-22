from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragk8s import pipeline, retrieval_metrics
from ragk8s.pipeline import RunConfig
from ragk8s.retrieval_metrics import question_metrics, source_rank


def test_source_rank() -> None:
    assert source_rank(["d1", "d2", "d2"], "d2") == 2
    assert source_rank(["d1"], "d9") is None


def _item(qid: str, docs: list[str], source: str = "d", category: str = "factual") -> dict[str, object]:
    return {
        "id": qid,
        "category": category,
        "source_doc_id": source,
        "hits": [{"doc_id": d, "text": "x" * 10} for d in docs],
    }


def test_question_metrics() -> None:
    m = question_metrics(_item("q", ["a", "b", "d", "c", "e"]))
    assert m == {"hit@1": 0.0, "hit@3": 1.0, "hit@5": 1.0, "mrr": pytest.approx(1 / 3), "context_chars": 50.0}
    assert question_metrics(_item("q", ["a", "b"]))["mrr"] == 0.0


def test_evaluate_ranks_configs_and_ignores_impossible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "RETRIEVAL_DIR", tmp_path)
    good, bad = RunConfig("structure", "hybrid_rerank"), RunConfig("fixed", "bm25")
    data = {
        good: [_item("q1", ["d"]), _item("q2", ["d"]), _item("q3", ["x"], source="", category="impossible")],
        bad: [_item("q1", ["x", "d"]), _item("q2", ["x"]), _item("q3", ["x"], source="", category="impossible")],
    }
    for cfg, items in data.items():
        pipeline.retrieval_path(cfg).write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    rows = retrieval_metrics.evaluate([bad, good], reference=good)
    assert [r["config"] for r in rows] == ["structure_hybrid_rerank", "fixed_bm25"]
    assert rows[0]["n"] == 2 and rows[0]["mrr"] == 1.0
    assert rows[1]["mrr"] == pytest.approx(0.25)
    assert rows[1]["mrr_diff_vs_ref"] == pytest.approx(-0.75)
