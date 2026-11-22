from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from openpyxl import load_workbook

from ragk8s import validation
from ragk8s.validation import agreement, cohen_kappa, sample, spearman


def test_spearman_and_kappa() -> None:
    assert spearman([0, 0.5, 1, 1], [0.1, 0.4, 0.9, 0.8]) == pytest.approx(0.9486, abs=1e-3)
    assert math.isnan(spearman([1, 1, 1], [0, 0.5, 1]))
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [1, 1, 0, 0]) == 0.0


def test_agreement_reports_judge_leniency() -> None:
    a = agreement([0, 0.5, 1, 1, 0, 1], [0.5, 1, 1, 1, 0.5, 1])
    assert a["bias_judge_minus_human"] == pytest.approx(0.25)
    assert a["mae"] == pytest.approx(0.25) and a["n"] == 6


def _items() -> list[dict[str, Any]]:
    items = []
    for model in ("big", "small"):
        for i in range(30):
            faith = 1.0 if i % 3 else 0.4
            items.append(
                {
                    "model": model,
                    "config": "fixed_bm25",
                    "id": f"q{i}",
                    "record": {},
                    "judge": {"faithfulness": faith, "answer_relevancy": 0.9, "context_recall": 1.0},
                }
            )
    return items


def test_sample_is_seeded_balanced_by_model_and_score() -> None:
    picked = sample(_items(), 20, seed=3)
    assert picked == sample(_items(), 20, seed=3)
    assert sum(i["model"] == "big" for i in picked) == 10
    assert sum(i["judge"]["faithfulness"] < 1 for i in picked) == 10


def _fixture(tmp: Path) -> tuple[Path, Path]:
    runs, scores = tmp / "runs" / "m", tmp / "judge" / "m"
    runs.mkdir(parents=True)
    scores.mkdir(parents=True)
    recs, rows = [], []
    for i in range(6):
        recs.append(
            {
                "id": f"q{i}",
                "question": f"Q{i}?",
                "answer": "A [1].",
                "contexts": ["c"],
                "ground_truth": "A",
                "category": "factual",
                "is_refusal": False,
            }
        )
        row: dict[str, Any] = {"id": f"q{i}"}
        for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            row[m], row[f"{m}_status"] = (1.0 if i % 2 else 0.0), "ok"
        rows.append(row)
    (runs / "run_fixed_bm25.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    (scores / "scores_fixed_bm25.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return tmp / "runs", tmp / "judge"


def test_export_hides_scores_then_agree(tmp_path: Path) -> None:
    runs, scores = _fixture(tmp_path)
    sheet = tmp_path / "v.xlsx"
    validation.export(n=6, path=sheet, scores_dir=scores, runs_dir=runs)
    wb = load_workbook(sheet)
    header = [c.value for c in wb["labels"][1]]
    assert "faithfulness" not in header and set(validation.PAIRS) <= set(header)
    with pytest.raises(ValueError, match="unlabelled"):
        validation.agree(sheet, scores_dir=scores, out=tmp_path / "a.json")

    keys = {str(r[0]): r[3] for r in list(wb["key"].iter_rows(values_only=True))[1:]}
    for row in wb["labels"].iter_rows(min_row=2):
        judge_value = 1.0 if int(keys[str(row[0].value)][1:]) % 2 else 0.0
        row[7].value = row[8].value = row[9].value = judge_value  # a human who fully agrees
    wb.save(sheet)
    result = validation.agree(sheet, scores_dir=scores, out=tmp_path / "a.json")
    faith = result["metrics"]["faithfulness"]
    assert faith["n"] == 6 and faith["mae"] == 0.0 and faith["kappa_binary_0.5"] == 1.0
    assert json.loads((tmp_path / "a.json").read_text())["n_items"] == 6
