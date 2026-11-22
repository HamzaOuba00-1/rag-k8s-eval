from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragk8s import compare
from ragk8s.judge import METRICS
from ragk8s.pipeline import RunConfig

BEST, OTHER = RunConfig("fixed", "hybrid_rerank"), RunConfig("fixed", "bm25")


def _write(results: Path, model: str, cfg: RunConfig, faith: float, coverage: float) -> None:
    run_dir, judge_dir = results / "runs" / model, results / "judge" / model
    run_dir.mkdir(parents=True, exist_ok=True)
    judge_dir.mkdir(parents=True, exist_ok=True)
    recs = [
        {
            "id": f"q{i}",
            "category": "factual",
            "is_refusal": i == 0,
            "citation_coverage": coverage,
            "invalid_citations": [],
        }
        for i in range(10)
    ]
    recs += [
        {"id": f"x{i}", "category": "impossible", "is_refusal": True, "citation_coverage": 0.0, "invalid_citations": []}
        for i in range(4)
    ]
    (run_dir / f"run_{cfg.name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    meta = {
        "complete": True,
        "tokens_mean": 1400.0,
        "generation_seconds_mean": 0.5,
        "created_at": "2026-11-22T00:00:00+00:00",
        "provider": "groq",
        "model": model,
    }
    (run_dir / f"run_{cfg.name}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    rows = []
    for r in recs[:10]:
        row = {"id": r["id"]}
        for m in METRICS:
            refused = r["is_refusal"] and m in ("faithfulness", "answer_relevancy")
            row[m] = None if refused else faith
            row[f"{m}_status"] = "n/a" if refused else "ok"
        rows.append(row)
    (judge_dir / f"scores_{cfg.name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (judge_dir / f"scores_{cfg.name}.meta.json").write_text(json.dumps({"judge": {"model": "gemma"}}), encoding="utf-8")


def test_build_tables_summary_and_figures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compare, "RESULTS", tmp_path)
    for name, value in (("OUT_MD", "comparison.md"), ("OUT_CSV", "comparison.csv"), ("SUMMARY", "summary.json")):
        monkeypatch.setattr(compare, name, tmp_path / value)
    monkeypatch.setattr(compare, "FIGURES", tmp_path / "figures")
    _write(tmp_path, compare.APP_MODEL, BEST, 0.9, 0.9)
    _write(tmp_path, compare.APP_MODEL, OTHER, 0.6, 0.8)
    _write(tmp_path, compare.REF_MODEL, BEST, 0.7, 0.6)

    summary = compare.build([BEST, OTHER, RunConfig("semantic", "dense")], BEST)
    assert [r["config"] for r in summary["configs"]] == ["fixed_hybrid_rerank", "fixed_bm25"]  # missing run skipped
    best = summary["configs"][0]
    assert best["faithfulness"]["n"] == 9 and best["context_recall"]["n"] == 10
    assert best["judge_complete"]
    assert best["refusal_rate"]["mean"] == 1.0 and best["false_refusal_rate"]["mean"] == pytest.approx(0.1)
    assert summary["paired_vs_best"]["fixed_bm25"]["faithfulness"]["diff"] == pytest.approx(-0.3)
    assert summary["app_vs_reference"]["faithfulness"]["diff"] == pytest.approx(0.2)
    md = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "0.90 [0.90, 0.90]" in md and "+0.200" in md
    assert (tmp_path / "figures" / "ragas_by_config.png").exists()
    assert (tmp_path / "figures" / "app_vs_reference.png").exists()
