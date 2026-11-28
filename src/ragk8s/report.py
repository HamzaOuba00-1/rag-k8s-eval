"""Fill the result sections of README.md and REPORT.md from the result files.

Numbers are never typed into the documentation: each block between
``<!-- BEGIN:<name> -->`` and ``<!-- END:<name> -->`` is regenerated from
results/retrieval_metrics.csv, results/summary.json and the run metadata. A block
whose source does not exist yet says so ("pending") instead of showing a number.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import config

RESULTS = config.RESULTS_DIR
DOCS = (Path("README.md"), Path("REPORT.md"))


def _ci(mean: float | None, low: float | None, high: float | None, digits: int = 2) -> str:
    if mean is None:
        return "pending"
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _est(e: dict[str, Any] | None) -> str:
    return _ci(e["mean"], e["low"], e["high"]) if e and e.get("n") else "pending"


def retrieval_block(limit: int | None = None) -> str:
    path = RESULTS / "retrieval_metrics.csv"
    if not path.exists():
        return "_Pending: run `python -m ragk8s retrieval-metrics`._"
    with path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("subset", "all") == "all"]
    rows.sort(key=lambda r: -float(r["mrr"]))
    meta = _retrieval_meta()
    lines = [
        f"Retrieval only, no LLM. {rows[0]['n']} factual questions; relevant = the question's source page is in the",
        f"context given to the generator ({config.CONTEXT_BUDGET_CHARS:,} characters for every strategy). "
        f"Mean with 95% bootstrap CI.{meta}",
        "",
        "| Chunking | Retrieval | MRR | Hit@1 | Hit@ctx | ΔMRR vs best (paired) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows[:limit] if limit else rows:
        diff = (
            f"{float(r['mrr_diff_vs_ref']):+.3f} [{float(r['mrr_diff_low']):+.3f}, {float(r['mrr_diff_high']):+.3f}]"
            if r.get("mrr_diff_vs_ref")
            else "reference"
        )
        lines.append(
            f"| {r['chunking']} | {r['retrieval']} | {_ci(float(r['mrr']), float(r['mrr_low']), float(r['mrr_high']))} | "
            f"{float(r['hit@1']):.2f} | {float(r['hit@ctx']):.2f} | {diff} |"
        )
    if limit and len(rows) > limit:
        lines += [
            "",
            f"Top {limit} of {len(rows)} configurations; full tables in [`results/retrieval_metrics.md`](results/retrieval_metrics.md).",
        ]
    return "\n".join(lines)


def _retrieval_meta() -> str:
    metas = sorted((RESULTS / "retrieval").glob("*.meta.json"))
    if not metas:
        return ""
    created = sorted(json.loads(p.read_text(encoding="utf-8")).get("created_at", "") for p in metas)
    return f" Computed {created[-1][:10]}."


def _summary() -> dict[str, Any] | None:
    path = RESULTS / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def generation_block() -> str:
    s = _summary()
    if not s or not s["configs"] or not all(r.get("judge_complete") for r in s["configs"]):
        return "_Pending: all runs judged by RAGAS, then `python -m ragk8s compare`._"
    lines = [
        f"Generator `{s['app_model']}`; judge `{(s.get('judge') or {}).get('model', 'pending')}`. "
        "Mean with 95% bootstrap CI over questions.",
        "",
        "| Chunking | Retrieval | Faithfulness | Answer relevancy | Context precision | Context recall | "
        "Refused out-of-corpus | False refusals |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in s["configs"]:
        lines.append(
            f"| {r['chunking']} | {r['retrieval']} | {_est(r['faithfulness'])} | {_est(r['answer_relevancy'])} | "
            f"{_est(r['context_precision'])} | {_est(r['context_recall'])} | {_est(r['refusal_rate'])} | "
            f"{_est(r['false_refusal_rate'])} |"
        )
    lines += ["", "Full tables and paired differences: [`results/comparison.md`](results/comparison.md)."]
    return "\n".join(lines)


def models_block() -> str:
    s = _summary()
    if not s or not s.get("reference") or not s["reference"].get("judge_complete"):
        return "_Pending: reference generator run fully judged, then `python -m ragk8s compare`._"
    app = next((r for r in s["configs"] if r["config"] == s["best_config"]), None)
    ref = s["reference"]
    lines = [
        f"Same contexts (`{s['best_config']}`), same prompt, same judge. Δ = paired bootstrap difference (* = CI excludes 0).",
        "",
        f"| Metric | {s['app_model']} | {s['reference_model']} | Δ |",
        "|---|---|---|---|",
    ]
    for m in ("faithfulness", "answer_relevancy", "citation_coverage"):
        d = s["app_vs_reference"].get(m)
        diff = (
            f"{d['diff']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]{' *' if d['significant'] else ''}"
            if d
            else "pending"
        )
        lines.append(f"| {m.replace('_', ' ')} | {_est(app[m] if app else None)} | {_est(ref[m])} | {diff} |")
    lines.append(
        f"| seconds per answer (mean) | {app['seconds_mean'] if app else 'pending'} | {ref['seconds_mean']} | — |"
    )
    return "\n".join(lines)


def _num(value: float, fmt: str = ".2f") -> str:
    return "n/a" if value != value else format(value, fmt)  # NaN: a side is constant


def agreement_block() -> str:
    path = RESULTS / "judge_agreement.json"
    if not path.exists():
        return "_Pending: judge validation labels (`python -m ragk8s validation export`, then `agree`)._"
    data = json.loads(path.read_text(encoding="utf-8"))
    annotator = data.get("annotator", "human")
    if annotator == "human":
        who = f"{data['n_items']} answers labelled blind by a human (0 / 0.5 / 1)."
    else:
        who = (
            f"{data['n_items']} answers labelled by an LLM ({annotator}, 0 / 0.5 / 1) without seeing the judge's "
            "scores: this is agreement between two models, not a human validation."
        )
    lines = [
        f"{who} Bias > 0 = judge more lenient than the annotator. n/a = one side gives the same value to every answer.",
        "",
        "| Metric | Spearman [95% CI] | MAE | Bias (judge - annotator) | Cohen's κ (≥ 0.5) | Mean, annotator / judge |",
        "|---|---|---|---|---|---|",
    ]
    for metric, a in data["metrics"].items():
        lines.append(
            f"| {metric} | {_num(a['spearman'])} [{_num(a['spearman_ci'][0])}, {_num(a['spearman_ci'][1])}] | "
            f"{a['mae']:.2f} | {a['bias_judge_minus_human']:+.2f} [{a['bias_ci'][0]:+.2f}, {a['bias_ci'][1]:+.2f}] | "
            f"{_num(a['kappa_binary_0.5'])} | {a['human_mean']:.2f} / {a['judge_mean']:.2f} |"
        )
    return "\n".join(lines)


def runs_block() -> str:
    metas = sorted((RESULTS / "runs").glob("*/run_*.meta.json"))
    if not metas:
        return "_Pending: generation runs._"
    lines = [
        "| Model | Configuration | Provider | Model digest | Questions | Tokens / answer | s / answer | Code commit | Date (UTC) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in metas:
        m = json.loads(p.read_text(encoding="utf-8"))
        digest = (m.get("model_digest") or "—")[:12]
        raw = m.get("git_commit")
        commit = f"{raw[:7]}{' (dirty)' if raw.endswith('-dirty') else ''}" if raw else "not recorded"
        lines.append(
            f"| {m['model']} | {m['config']} | {m['provider']} | {digest} | {m['n_questions']}"
            f"{'' if m.get('complete') else ' (partial)'} | {m.get('tokens_mean')} | {m.get('generation_seconds_mean')} | "
            f"{commit} | {m['created_at'][:16].replace('T', ' ')} |"
        )
    return "\n".join(lines)


BLOCKS: dict[str, Callable[[], str]] = {
    "retrieval-top": lambda: retrieval_block(limit=5),
    "retrieval": retrieval_block,
    "generation": generation_block,
    "models": models_block,
    "agreement": agreement_block,
    "runs": runs_block,
}


def fill(text: str) -> tuple[str, list[str]]:
    """Replace every known block in ``text``; returns the new text and the blocks filled."""
    filled = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in BLOCKS:
            return match.group(0)
        filled.append(name)
        return f"<!-- BEGIN:{name} -->\n{BLOCKS[name]()}\n<!-- END:{name} -->"

    pattern = re.compile(r"<!-- BEGIN:([\w-]+) -->.*?<!-- END:\1 -->", re.S)
    return pattern.sub(replace, text), filled


def run(paths: tuple[Path, ...] = DOCS) -> dict[str, list[str]]:
    done = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        new, filled = fill(text)
        if new != text:
            path.write_text(new, encoding="utf-8", newline="\n")
        done[str(path)] = filled
    return done
