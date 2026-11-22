from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ragk8s import report


def test_blocks_say_pending_without_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    text = (
        "intro\n<!-- BEGIN:generation -->\nold 0.99\n<!-- END:generation -->\n"
        "<!-- BEGIN:unknown -->x<!-- END:unknown -->"
    )
    new, filled = report.fill(text)
    assert filled == ["generation"]
    assert "old 0.99" not in new and "_Pending" in new and "<!-- BEGIN:unknown -->x" in new


def test_retrieval_and_runs_blocks_come_from_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    fields = [
        "subset",
        "config",
        "chunking",
        "retrieval",
        "n",
        "mrr",
        "mrr_low",
        "mrr_high",
        "hit@1",
        "hit@ctx",
        "mrr_diff_vs_ref",
        "mrr_diff_low",
        "mrr_diff_high",
    ]
    rows = [
        ["all", "fixed_bm25", "fixed", "bm25", "45", "0.70", "0.60", "0.80", "0.6", "0.9", "-0.1", "-0.2", "0.01"],
        [
            "all",
            "fixed_hybrid_rerank",
            "fixed",
            "hybrid_rerank",
            "45",
            "0.84",
            "0.75",
            "0.92",
            "0.76",
            "0.96",
            "",
            "",
            "",
        ],
        ["page", "fixed_bm25", "fixed", "bm25", "18", "0.10", "0.0", "0.2", "0.1", "0.2", "", "", ""],
    ]
    with (tmp_path / "retrieval_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)
    block = report.retrieval_block()
    lines = [line for line in block.splitlines() if line.startswith("| fixed")]
    assert lines[0].startswith("| fixed | hybrid_rerank | 0.84 [0.75, 0.92]") and "reference" in lines[0]
    assert "-0.100 [-0.200, +0.010]" in lines[1] and len(lines) == 2  # page subset excluded

    run_dir = tmp_path / "runs" / "m"
    run_dir.mkdir(parents=True)
    meta = {
        "model": "m",
        "config": "fixed_bm25",
        "provider": "groq",
        "model_digest": None,
        "n_questions": 53,
        "complete": False,
        "tokens_mean": 1400.0,
        "generation_seconds_mean": 0.5,
        "git_commit": "1368703abcdef-dirty",
        "created_at": "2026-11-22T12:00:00+00:00",
    }
    (run_dir / "run_fixed_bm25.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    runs = report.runs_block()
    assert "53 (partial)" in runs and "1368703 (dirty)" in runs and "2026-11-22 12:00" in runs


def test_partial_judging_is_not_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    est = {"mean": 0.9, "low": 0.8, "high": 1.0, "n": 9}
    row = {"chunking": "fixed", "retrieval": "bm25", "config": "fixed_bm25", "judge_complete": False}
    row.update(
        {
            m: est
            for m in (
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "refusal_rate",
                "false_refusal_rate",
            )
        }
    )
    summary = {"app_model": "m", "configs": [row], "judge": {"model": "j"}}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert report.generation_block().startswith("_Pending")
    row["judge_complete"] = True
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert "0.90 [0.80, 1.00]" in report.generation_block()
