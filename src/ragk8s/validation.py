"""Judge validation: agreement between the RAGAS judge and a human on a sample.

export  draws a stratified, seeded sample of judged answers (both generators, low
        and high judge scores) into eval/review/judge_validation.xlsx. The judge's
        scores are NOT shown, so the human labels are blind.
agree   reads the human labels and compares them with the judge:
        Spearman correlation (bootstrap CI), mean absolute error, mean bias
        (judge - human: positive = lenient judge) and Cohen's kappa after binarising
        both sides at 0.5.

Human labels use three levels, matching the RAGAS metrics:
  supported         every claim of the answer is in the contexts (1 / 0.5 / 0)  ~ faithfulness
  relevant          the answer addresses the question (1 / 0.5 / 0)              ~ answer_relevancy
  context_covers    the contexts contain the reference answer (1 / 0.5 / 0)      ~ context_recall
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from . import config
from .stats import bootstrap_ci
from .utils import read_jsonl, write_json

SHEET_PATH = config.EVAL_DIR / "review" / "judge_validation.xlsx"
OUT_PATH = config.RESULTS_DIR / "judge_agreement.json"
PAIRS = {"supported": "faithfulness", "relevant": "answer_relevancy", "context_covers": "context_recall"}
LEVELS = (0.0, 0.5, 1.0)
COLUMNS = ("item", "model", "config", "question", "answer", "contexts", "reference", *PAIRS, "notes")

INSTRUCTIONS = [
    "Judge validation - rate each answer yourself; the judge's scores are hidden.",
    "",
    "supported:      1 = every claim of the answer is stated in the contexts; 0.5 = some claims are not; "
    "0 = the main claim is not supported.",
    "relevant:       1 = the answer directly addresses the question; 0.5 = partly or with off-topic content; 0 = no.",
    "context_covers: 1 = the contexts contain everything in the reference answer; 0.5 = part of it; 0 = none.",
    "",
    "Judge the answer against the CONTEXTS shown (not your own knowledge). Use 'notes' when unsure.",
    "Do not edit the 'item' column. Save as .xlsx when done.",
]


def _judged_items(scores_dir: Path, runs_dir: Path) -> list[dict[str, Any]]:
    items = []
    for scores_path in sorted(scores_dir.glob("*/scores_*.jsonl")):
        model, cfg = scores_path.parent.name, scores_path.stem.removeprefix("scores_")
        run = {r["id"]: r for r in read_jsonl(runs_dir / model / f"run_{cfg}.jsonl")}
        for row in read_jsonl(scores_path):
            if any(row.get(f"{m}_status") != "ok" for m in PAIRS.values()):
                continue
            record = run[row["id"]]
            items.append(
                {
                    "model": model,
                    "config": cfg,
                    "id": row["id"],
                    "record": record,
                    "judge": {m: row[m] for m in PAIRS.values()},
                }
            )
    return items


def sample(items: Sequence[dict[str, Any]], n: int, seed: int = config.SEED) -> list[dict[str, Any]]:
    """Seeded sample stratified by model and by judge faithfulness (< 1 vs = 1).

    Low-faithfulness answers are rarer but are where a judge matters most, so each
    model's quota is split evenly between the two bins when possible.
    """
    rng = random.Random(seed)
    models = sorted({i["model"] for i in items})
    picked: list[dict[str, Any]] = []
    for k, model in enumerate(models):
        quota = n // len(models) + (1 if k < n % len(models) else 0)
        pool = [i for i in items if i["model"] == model]
        low = [i for i in pool if (i["judge"]["faithfulness"] or 0) < 1]
        high = [i for i in pool if (i["judge"]["faithfulness"] or 0) >= 1]
        take_low = min(len(low), quota // 2)
        take_high = min(len(high), quota - take_low)
        take_low = min(len(low), quota - take_high)
        picked += rng.sample(low, take_low) + rng.sample(high, take_high)
    rng.shuffle(picked)
    return picked


def export(
    n: int = 40,
    path: Path = SHEET_PATH,
    scores_dir: Path = config.RESULTS_DIR / "judge",
    runs_dir: Path = config.RESULTS_DIR / "runs",
    overwrite: bool = False,
) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (it may contain your labels). Use --overwrite to replace it.")
    chosen = sample(_judged_items(scores_dir, runs_dir), n)
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "labels"
    ws.append(list(COLUMNS))
    key: list[dict[str, str]] = []
    for idx, item in enumerate(chosen, start=1):
        rec = item["record"]
        contexts = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(rec["contexts"], start=1))
        ws.append(
            [
                idx,
                item["model"],
                item["config"],
                rec["question"],
                rec["answer"],
                contexts,
                rec["ground_truth"],
                None,
                None,
                None,
                None,
            ]
        )
        key.append({"item": str(idx), "model": item["model"], "config": item["config"], "id": item["id"]})
    for col, width in {
        "A": 6,
        "B": 14,
        "C": 18,
        "D": 40,
        "E": 60,
        "F": 100,
        "G": 40,
        "H": 11,
        "I": 11,
        "J": 14,
        "K": 30,
    }.items():
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "D2"
    dv = DataValidation(type="list", formula1='"0,0.5,1"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"H2:J{ws.max_row}")
    info = wb.create_sheet("instructions")
    for line in INSTRUCTIONS:
        info.append([line])
    info.column_dimensions["A"].width = 140
    keys = wb.create_sheet("key")  # links items to runs; contains no judge score
    keys.append(["item", "model", "config", "id"])
    for k in key:
        keys.append([k["item"], k["model"], k["config"], k["id"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


# --- agreement statistics -------------------------------------------------------------


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    x, y = _ranks(np.asarray(a, dtype=float)), _ranks(np.asarray(b, dtype=float))
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def cohen_kappa(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    labels = np.union1d(a_arr, b_arr)
    observed = float((a_arr == b_arr).mean())
    expected = float(sum((a_arr == k).mean() * (b_arr == k).mean() for k in labels))
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def spearman_ci(
    human: npt.ArrayLike, judge: npt.ArrayLike, n_resamples: int = 5000, seed: int = config.SEED
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    h, j = np.asarray(human, dtype=float), np.asarray(judge, dtype=float)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(h), len(h))
        rho = spearman(h[idx], j[idx])
        if not np.isnan(rho):
            values.append(rho)
    if not values:
        return float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def agreement(human: npt.ArrayLike, judge: npt.ArrayLike) -> dict[str, Any]:
    h, j = np.asarray(human, dtype=float), np.asarray(judge, dtype=float)
    low, high = spearman_ci(h, j)
    bias = bootstrap_ci(list(j - h))
    return {
        "n": len(h),
        "spearman": spearman(h, j),
        "spearman_ci": [low, high],
        "mae": float(np.abs(j - h).mean()),
        "bias_judge_minus_human": bias.value,
        "bias_ci": [bias.low, bias.high],
        "kappa_binary_0.5": cohen_kappa((h >= 0.5).astype(int), (j >= 0.5).astype(int)),
        "human_mean": float(h.mean()),
        "judge_mean": float(j.mean()),
    }


def agree(
    path: Path = SHEET_PATH, scores_dir: Path = config.RESULTS_DIR / "judge", out: Path = OUT_PATH
) -> dict[str, Any]:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True)
    rows = list(wb["labels"].iter_rows(values_only=True))
    header = [str(h) for h in rows[0]]
    labels = [dict(zip(header, r, strict=False)) for r in rows[1:] if r and r[0] is not None]
    keys = {
        str(r[0]): {"model": r[1], "config": r[2], "id": r[3]} for r in list(wb["key"].iter_rows(values_only=True))[1:]
    }
    judge_rows = {
        (p.parent.name, p.stem.removeprefix("scores_"), r["id"]): r
        for p in scores_dir.glob("*/scores_*.jsonl")
        for r in read_jsonl(p)
    }
    missing = [str(lab["item"]) for lab in labels if any(lab.get(c) in (None, "") for c in PAIRS)]
    if missing:
        raise ValueError(f"unlabelled items: {', '.join(missing)}")
    result: dict[str, Any] = {"n_items": len(labels), "sheet": str(path), "metrics": {}}
    for human_col, metric in PAIRS.items():
        h, j = [], []
        for lab in labels:
            k = keys[str(lab["item"])]
            score = judge_rows[(k["model"], k["config"], k["id"])][metric]
            value = float(lab[human_col])
            if value not in LEVELS:
                raise ValueError(f"item {lab['item']}: {human_col} must be 0, 0.5 or 1")
            h.append(value)
            j.append(float(score))
        result["metrics"][metric] = {"human_label": human_col, **agreement(h, j)}
    write_json(out, result)
    return result
