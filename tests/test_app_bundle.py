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
    assert report["judge_complete"] is False
    assert bundle["live_config"] == "fixed_hybrid" and bundle["best_config"] == "fixed_hybrid_rerank"


HEADER = (
    "subset,config,chunking,retrieval,n,hit@1,hit@1_low,hit@1_high,hit@3,hit@3_low,hit@3_high,hit@ctx,hit@ctx_low,"
    "hit@ctx_high,mrr,mrr_low,mrr_high,context_chunks,context_chunks_low,context_chunks_high,context_chars,"
    "context_chars_low,context_chars_high,mrr_diff_vs_ref,mrr_diff_low,mrr_diff_high,mrr_diff_p"
)


def test_retrieval_rows_follow_the_metrics_csv(tree: Path) -> None:
    results = tree / "results"
    (results / "retrieval_metrics.csv").write_text(
        HEADER
        + "\nall,fixed_hybrid_rerank,fixed,hybrid_rerank,45,0.76,0.62,0.87,0.9,0.8,0.96,0.93,0.84,1.0,0.84,0.75,0.92,"
        "4,3,5,4879,4700,5000,,,,"
        "\npage,fixed_dense,fixed,dense,18,0.5,0.3,0.7,0.6,0.4,0.8,0.7,0.5,0.9,0.6,0.4,0.8,4,3,5,4800,4600,5000,"
        "-0.2,-0.3,-0.1,0.001\n",
        encoding="utf-8",
    )
    out = tree / "bundle.json"
    app_bundle.build(out, results, results / "retrieval", results / "runs")
    rows = json.loads(out.read_text(encoding="utf-8"))["scores"]["retrieval"]
    assert [r["subset"] for r in rows] == ["all", "page"]
    assert rows[0]["mrr"] == {"mean": 0.84, "low": 0.75, "high": 0.92, "n": 45}
    assert rows[0]["hit_at_ctx"]["mean"] == 0.93 and rows[0]["mrr_diff_vs_best"] is None
    assert rows[1]["mrr_diff_vs_best"] == {"mean": -0.2, "low": -0.3, "high": -0.1, "n": 18}


def _row(config: str, complete: bool) -> dict[str, Any]:
    est = {"mean": 0.8, "low": 0.7, "high": 0.9, "n": 40}
    return {
        "config": config,
        "judge_complete": complete,
        "citation_coverage": est,
        "faithfulness": est,
        "answer_relevancy": est,
        "context_precision": est,
        "context_recall": est,
    }


def test_judge_metrics_stay_null_until_the_judge_run_is_complete(tree: Path) -> None:
    results = tree / "results"
    out = tree / "bundle.json"
    write_json(
        results / "summary.json",
        {
            "configs": [_row("a", True), _row("b", False)],
            "reference": _row("a", True),
            "judge_agreement": {"spearman": 0.7},
        },
    )
    report = app_bundle.build(out, results, results / "retrieval", results / "runs")
    gen = json.loads(out.read_text(encoding="utf-8"))["scores"]["generation"]
    assert report["generation_summary"] is True and report["judge_complete"] is False
    assert gen["configs"][0]["faithfulness"] is not None
    assert gen["configs"][1]["faithfulness"] is None and gen["configs"][1]["citation_coverage"] is not None
    assert gen["judge_agreement"] is None

    agreement = {
        "n_items": 40,
        "annotator": "claude-opus-5-5",
        "metrics": {
            "faithfulness": {
                "n": 40, "spearman": -0.04, "spearman_ci": [-0.09, -0.03], "mae": 0.04,
                "bias_judge_minus_human": -0.01, "bias_ci": [-0.05, 0.03], "kappa_binary_0.5": float("nan"),
                "human_mean": 0.99, "judge_mean": 0.97,
            }
        },
    }
    write_json(
        results / "summary.json",
        {"configs": [_row("a", True)], "reference": _row("a", True), "judge_agreement": agreement},
    )
    assert app_bundle.build(out, results, results / "retrieval", results / "runs")["judge_complete"] is True
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text
    gen = json.loads(text)["scores"]["generation"]
    assert gen["judge_agreement"]["annotator"] == "claude-opus-5-5"
    row = gen["judge_agreement"]["metrics"][0]
    assert row["metric"] == "faithfulness" and row["bias"] == -0.01 and row["kappa"] is None
