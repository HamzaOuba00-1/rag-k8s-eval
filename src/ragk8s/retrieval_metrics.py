"""Retrieval metrics without any LLM: is the source document of the gold answer retrieved?

Each factual question was written from one chunk of one document
(``source_doc_id``). A retrieved chunk is counted as relevant when it comes
from that document. Matching at the document level makes the metric
comparable across chunking strategies, whose chunk ids differ.

Metrics per question: hit@1, hit@3, hit@k (k = number of retrieved chunks),
reciprocal rank (0 when the document is not retrieved) and the number of
context characters handed to the generator.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import config
from .pipeline import RETRIEVAL_DIR, RunConfig, retrieval_path
from .stats import Estimate, bootstrap_ci, paired_bootstrap
from .utils import read_jsonl

METRICS = ("hit@1", "hit@3", "hit@5", "mrr", "context_chars")
OUT_CSV = config.RESULTS_DIR / "retrieval_metrics.csv"
OUT_MD = config.RESULTS_DIR / "retrieval_metrics.md"
PER_QUESTION_CSV = config.RESULTS_DIR / "retrieval_per_question.csv"


def source_rank(hit_doc_ids: Sequence[str], source_doc_id: str) -> int | None:
    """1-based rank of the first chunk from ``source_doc_id``, or None."""
    for rank, doc_id in enumerate(hit_doc_ids, start=1):
        if doc_id == source_doc_id:
            return rank
    return None


def question_metrics(item: dict[str, Any]) -> dict[str, float]:
    rank = source_rank([h["doc_id"] for h in item["hits"]], item["source_doc_id"])
    return {
        "hit@1": float(rank is not None and rank <= 1),
        "hit@3": float(rank is not None and rank <= 3),
        "hit@5": float(rank is not None and rank <= 5),
        "mrr": 1.0 / rank if rank else 0.0,
        "context_chars": float(sum(len(h["text"]) for h in item["hits"])),
    }


def per_question(cfg: RunConfig) -> dict[str, dict[str, float]]:
    """Metrics of every factual question with a source document, keyed by question id."""
    return {
        item["id"]: question_metrics(item)
        for item in read_jsonl(retrieval_path(cfg))
        if item["category"] == "factual" and item.get("source_doc_id")
    }


def evaluate(configs: Sequence[RunConfig], reference: RunConfig | None = None) -> list[dict[str, Any]]:
    """Aggregate with bootstrap CIs; optionally a paired MRR difference vs ``reference``."""
    scores = {cfg: per_question(cfg) for cfg in configs}
    ids = sorted(set.intersection(*(set(s) for s in scores.values())))
    if not ids:
        raise ValueError("no common factual question across configurations")
    rows = []
    for cfg in configs:
        row: dict[str, Any] = {"config": cfg.name, "chunking": cfg.strategy, "retrieval": cfg.method, "n": len(ids)}
        for metric in METRICS:
            est: Estimate = bootstrap_ci([scores[cfg][i][metric] for i in ids])
            row[metric], row[f"{metric}_low"], row[f"{metric}_high"] = est.value, est.low, est.high
        if reference is not None and reference in scores and cfg != reference:
            d = paired_bootstrap([scores[cfg][i]["mrr"] for i in ids], [scores[reference][i]["mrr"] for i in ids])
            row.update(
                {"mrr_diff_vs_ref": d.diff, "mrr_diff_low": d.low, "mrr_diff_high": d.high, "mrr_diff_p": d.p_value}
            )
        rows.append(row)
    return sorted(rows, key=lambda r: -r["mrr"])


def write_reports(rows: list[dict[str, Any]], configs: Sequence[RunConfig], reference: RunConfig | None) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)

    ids = sorted(set.intersection(*(set(per_question(c)) for c in configs)))
    with PER_QUESTION_CSV.open("w", encoding="utf-8", newline="") as f:
        rows_writer = csv.writer(f)
        rows_writer.writerow(["question_id", "config", *METRICS])
        for cfg in configs:
            pq = per_question(cfg)
            for i in ids:
                rows_writer.writerow([i, cfg.name, *(pq[i][m] for m in METRICS)])

    ref = f" Paired difference of MRR vs `{reference.name}` (95% CI)." if reference else ""
    lines = [
        f"Retrieval metrics on {rows[0]['n']} factual questions (source document in the top-k).",
        f"Mean with 95% bootstrap CI ({len(ids)} questions, 10,000 resamples).{ref}",
        "",
        "| Chunking | Retrieval | Hit@1 | Hit@3 | Hit@5 | MRR | Context chars | ΔMRR vs ref |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cells = [r["chunking"], r["retrieval"]]
        cells += [f"{r[m]:.2f} [{r[m + '_low']:.2f}, {r[m + '_high']:.2f}]" for m in ("hit@1", "hit@3", "hit@5", "mrr")]
        cells.append(f"{r['context_chars']:.0f}")
        if "mrr_diff_vs_ref" in r:
            cells.append(f"{r['mrr_diff_vs_ref']:+.3f} [{r['mrr_diff_low']:+.3f}, {r['mrr_diff_high']:+.3f}]")
        else:
            cells.append("reference" if reference and r["config"] == reference.name else "")
        lines.append("| " + " | ".join(cells) + " |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run(configs: Sequence[RunConfig] | None = None, reference: RunConfig | None = None) -> Path:
    available = [c for c in (configs or _available_configs()) if retrieval_path(c).exists()]
    if not available:
        raise FileNotFoundError(f"no retrieval files in {RETRIEVAL_DIR}: run `python -m ragk8s retrieve --all` first")
    rows = evaluate(available, reference)
    if reference is None:
        reference = next(c for c in available if c.name == rows[0]["config"])
        rows = evaluate(available, reference)
    write_reports(rows, available, reference)
    print(OUT_MD.read_text(encoding="utf-8"))
    return OUT_MD


def _available_configs() -> list[RunConfig]:
    from .pipeline import ALL_CONFIGS

    return list(ALL_CONFIGS)
