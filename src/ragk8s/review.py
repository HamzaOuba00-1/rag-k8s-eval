"""Human review of the gold questions (export to Excel, apply the verdicts back).

The factual questions were drafted by a local LLM from single chunks. Before
they are used as a gold standard, each one is checked by a human:

    python -m ragk8s review export   -> eval/review/questions_review.xlsx
    (fill the "verdict" column: keep / fix / drop)
    python -m ragk8s review apply    -> updates eval/questions.jsonl

The system's answers are deliberately not shown, so they cannot bias the gold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import config
from .textclean import clean_markup
from .utils import read_jsonl, write_jsonl

REVIEW_PATH = config.EVAL_DIR / "review" / "questions_review.xlsx"
QUESTIONS_PATH = config.EVAL_DIR / "questions.jsonl"
DROPPED_PATH = config.EVAL_DIR / "questions_dropped.jsonl"
VERDICTS = ("keep", "fix", "drop")
COLUMNS = (
    "row",
    "question",
    "ground_truth",
    "source_excerpt",
    "source_title",
    "source_url",
    "verdict",
    "corrected_question",
    "corrected_ground_truth",
    "notes",
)

INSTRUCTIONS = [
    "Review of the gold questions — one row per factual question.",
    "",
    "For each row, read the source excerpt, then set 'verdict':",
    "  keep  the question is clear, answerable from the Kubernetes docs, and the ground truth is correct and complete;",
    "  fix   the idea is good but the wording or the ground truth must change: fill corrected_question and/or "
    "corrected_ground_truth (leave a cell empty to keep the original);",
    "  drop  the question is ambiguous, trivial, unanswerable, or depends on the excerpt itself "
    "('in this section...').",
    "",
    "Checks: one unambiguous answer; the ground truth only states what the excerpt says; "
    "no reference to 'the excerpt' or 'this page'; English.",
    "Use 'notes' for anything worth reporting (e.g. why a question was dropped).",
    "Do not change the 'row' column. Save the file as .xlsx when done.",
]


def export(path: Path = REVIEW_PATH, questions_path: Path = QUESTIONS_PATH, overwrite: bool = False) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (it may contain your review). Use --overwrite to replace it.")
    questions = read_jsonl(questions_path)
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "review"
    ws.append(list(COLUMNS))
    for i, q in enumerate(questions):
        if q.get("category", "factual") != "factual":
            continue
        excerpt = clean_markup(q.get("source_chunk_text") or "")
        ws.append(
            [i, q["question"], q["ground_truth"], excerpt, q.get("source_title"), q.get("source_url"), "", "", "", ""]
        )

    widths = {"A": 6, "B": 50, "C": 50, "D": 90, "E": 25, "F": 30, "G": 10, "H": 40, "I": 40, "J": 30}
    for col, width in widths.items():
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "B2"
    dv = DataValidation(type="list", formula1='"' + ",".join(VERDICTS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"G2:G{ws.max_row}")

    info = wb.create_sheet("instructions")
    for line in INSTRUCTIONS:
        info.append([line])
    info.column_dimensions["A"].width = 140

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


def read_verdicts(path: Path = REVIEW_PATH) -> list[dict[str, Any]]:
    from openpyxl import load_workbook

    ws = load_workbook(path, read_only=True)["review"]
    rows = ws.iter_rows(values_only=True)
    header = [str(h) for h in next(rows)]
    if tuple(header[: len(COLUMNS)]) != COLUMNS:
        raise ValueError(f"{path}: unexpected columns {header}")
    out = []
    for values in rows:
        record = dict(zip(header, values, strict=False))
        if record.get("row") is None:
            continue
        record = {k: (v.strip() if isinstance(v, str) else v) for k, v in record.items()}
        record["verdict"] = (record.get("verdict") or "").lower()
        out.append(record)
    return out


def apply(
    path: Path = REVIEW_PATH,
    questions_path: Path = QUESTIONS_PATH,
    dropped_path: Path = DROPPED_PATH,
) -> dict[str, int]:
    """Apply verdicts. Refuses to write anything if a verdict is missing or invalid."""
    questions = read_jsonl(questions_path)
    verdicts = read_verdicts(path)
    problems = [f"row {v['row']}: verdict {v['verdict']!r}" for v in verdicts if v["verdict"] not in VERDICTS]
    problems += [
        f"row {v['row']}: 'fix' without any correction"
        for v in verdicts
        if v["verdict"] == "fix" and not (v.get("corrected_question") or v.get("corrected_ground_truth"))
    ]
    by_row = {int(v["row"]): v for v in verdicts}
    expected = {i for i, q in enumerate(questions) if q.get("category", "factual") == "factual"}
    if set(by_row) != expected:
        problems.append(f"rows do not match {questions_path}: re-export the review file")
    if problems:
        raise ValueError("review incomplete:\n  " + "\n  ".join(problems))

    kept, dropped = [], []
    counts = dict.fromkeys(VERDICTS, 0)
    for i, q in enumerate(questions):
        v = by_row.get(i)
        if v is None:  # impossible questions are reviewed separately
            kept.append(q)
            continue
        counts[v["verdict"]] += 1
        q = {**q, "reviewed": True}
        if v.get("notes"):
            q["review_notes"] = v["notes"]
        if v["verdict"] == "drop":
            dropped.append(q)
            continue
        if v["verdict"] == "fix":
            q["original_question"], q["original_ground_truth"] = q["question"], q["ground_truth"]
            q["question"] = v.get("corrected_question") or q["question"]
            q["ground_truth"] = v.get("corrected_ground_truth") or q["ground_truth"]
        kept.append(q)
    write_jsonl(questions_path, kept)
    if dropped:
        write_jsonl(dropped_path, read_jsonl(dropped_path) + dropped if dropped_path.exists() else dropped)
    return counts
