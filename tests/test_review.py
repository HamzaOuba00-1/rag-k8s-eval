from __future__ import annotations

import json
from pathlib import Path

import pytest
from openpyxl import load_workbook

from ragk8s import review


def _questions(path: Path) -> None:
    rows = [
        {"question": "Q1?", "ground_truth": "A1", "category": "factual", "source_chunk_text": "<!-- x -->Text 1"},
        {"question": "Docker Swarm?", "ground_truth": "n/a", "category": "impossible"},
        {"question": "Q2?", "ground_truth": "A2", "category": "factual", "source_chunk_text": "Text 2"},
        {"question": "Q3?", "ground_truth": "A3", "category": "factual", "source_chunk_text": "Text 3"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _fill(path: Path, verdicts: dict[int, tuple[str, str, str]]) -> None:
    wb = load_workbook(path)
    ws = wb["review"]
    for row in ws.iter_rows(min_row=2):
        if row[0].value in verdicts:
            row[6].value, row[7].value, row[8].value = verdicts[row[0].value]
    wb.save(path)


def test_export_then_apply(tmp_path: Path) -> None:
    qpath, xlsx, dropped = tmp_path / "q.jsonl", tmp_path / "review.xlsx", tmp_path / "dropped.jsonl"
    _questions(qpath)
    review.export(xlsx, qpath)
    rows = review.read_verdicts(xlsx)
    assert [r["row"] for r in rows] == [0, 2, 3]  # impossible question not exported
    assert rows[0]["source_excerpt"] == "Text 1"  # markup cleaned
    with pytest.raises(FileExistsError):
        review.export(xlsx, qpath)

    _fill(xlsx, {0: ("Keep", "", ""), 2: ("fix", "Better Q2?", ""), 3: ("drop", "", "")})
    counts = review.apply(xlsx, qpath, dropped)
    assert counts == {"keep": 1, "fix": 1, "drop": 1}
    out = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines()]
    assert [q["question"] for q in out] == ["Q1?", "Docker Swarm?", "Better Q2?"]
    assert out[2]["original_question"] == "Q2?" and out[2]["ground_truth"] == "A2" and out[2]["reviewed"]
    assert json.loads(dropped.read_text(encoding="utf-8"))["question"] == "Q3?"


def test_apply_refuses_incomplete_review(tmp_path: Path) -> None:
    qpath, xlsx = tmp_path / "q.jsonl", tmp_path / "review.xlsx"
    _questions(qpath)
    review.export(xlsx, qpath)
    _fill(xlsx, {0: ("keep", "", ""), 2: ("fix", "", "")})
    before = qpath.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="row 3"):
        review.apply(xlsx, qpath, tmp_path / "d.jsonl")
    assert qpath.read_text(encoding="utf-8") == before
