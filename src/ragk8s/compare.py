"""Final comparison tables, figures and the machine-readable summary.

Everything is computed from the files written by the pipeline (results/runs,
results/judge, results/judge_agreement.json, results/retrieval_metrics.csv), never
typed by hand:

- results/comparison.md / .csv   per configuration (application generator) and
                                  application vs reference generator on the best one
- results/figures/*.png           metric means with 95% bootstrap intervals
- results/summary.json            the same numbers for the web app's Scores page

Means come with 95% bootstrap intervals over questions; comparisons between two
configurations or two generators use paired bootstrap differences on the questions
scored in both.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import config
from .judge import METRICS
from .pipeline import BEST_CONFIG, MAIN_CONFIGS, RunConfig
from .stats import bootstrap_ci, paired_bootstrap
from .utils import read_jsonl, write_json

RESULTS = config.RESULTS_DIR
APP_MODEL = config.GROQ_MODEL.split("/")[-1]
REF_MODEL = config.OLLAMA_MODEL.replace(":", "-")
OUT_MD = RESULTS / "comparison.md"
OUT_CSV = RESULTS / "comparison.csv"
SUMMARY = RESULTS / "summary.json"
FIGURES = RESULTS / "figures"


def _load(
    model: str, cfg: RunConfig
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]] | None:
    run_path = RESULTS / "runs" / model / f"run_{cfg.name}.jsonl"
    meta_path = run_path.with_suffix(".meta.json")
    if not run_path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("complete"):
        return None
    records = {r["id"]: r for r in read_jsonl(run_path)}
    scores_path = RESULTS / "judge" / model / f"scores_{cfg.name}.jsonl"
    scores = {r["id"]: r for r in read_jsonl(scores_path)} if scores_path.exists() else {}
    return records, meta, scores


def _metric_values(scores: dict[str, dict[str, Any]], metric: str) -> dict[str, float]:
    return {qid: float(r[metric]) for qid, r in scores.items() if r.get(f"{metric}_status") == "ok"}


def _est(values: Sequence[float]) -> dict[str, Any]:
    e = bootstrap_ci(list(values))
    return {"mean": e.value, "low": e.low, "high": e.high, "n": e.n}


def config_summary(model: str, cfg: RunConfig) -> dict[str, Any] | None:
    loaded = _load(model, cfg)
    if loaded is None:
        return None
    records, meta, scores = loaded
    factual = [r for r in records.values() if r["category"] == "factual"]
    impossible = [r for r in records.values() if r["category"] == "impossible"]
    answered = [r for r in factual if not r["is_refusal"]]
    out: dict[str, Any] = {
        "model": model,
        "config": cfg.name,
        "chunking": cfg.strategy,
        "retrieval": cfg.method,
        "n_factual": len(factual),
        "n_impossible": len(impossible),
        "refusal_rate": _est([float(r["is_refusal"]) for r in impossible]),
        "false_refusal_rate": _est([float(r["is_refusal"]) for r in factual]),
        "citation_coverage": _est([r["citation_coverage"] for r in answered]),
        "answers_with_invalid_citation": sum(1 for r in records.values() if r["invalid_citations"]),
        "tokens_mean": meta.get("tokens_mean"),
        "seconds_mean": meta.get("generation_seconds_mean"),
        "generated_at": meta.get("created_at"),
        "provider": meta.get("provider"),
        "model_id": meta.get("model"),
        "judged": bool(scores),
    }
    for metric in METRICS:
        values = _metric_values(scores, metric)
        out[metric] = _est(list(values.values())) if values else None
    # Complete = every factual question has its context scores and every answered one its answer scores.
    expected = {
        "faithfulness": len(answered),
        "answer_relevancy": len(answered),
        "context_precision": len(factual),
        "context_recall": len(factual),
    }
    out["judge_complete"] = bool(scores) and all((out[m] or {}).get("n", 0) == n for m, n in expected.items())
    return out


def paired(model_a: str, cfg_a: RunConfig, model_b: str, cfg_b: RunConfig, metric: str) -> dict[str, Any] | None:
    a, b = _load(model_a, cfg_a), _load(model_b, cfg_b)
    if a is None or b is None:
        return None
    if metric in METRICS:
        va, vb = _metric_values(a[2], metric), _metric_values(b[2], metric)
    else:  # citation_coverage on questions both answered
        va = {q: r["citation_coverage"] for q, r in a[0].items() if r["category"] == "factual" and not r["is_refusal"]}
        vb = {q: r["citation_coverage"] for q, r in b[0].items() if r["category"] == "factual" and not r["is_refusal"]}
    ids = sorted(set(va) & set(vb))
    if not ids:
        return None
    d = paired_bootstrap([va[i] for i in ids], [vb[i] for i in ids])
    return {"diff": d.diff, "low": d.low, "high": d.high, "p_value": d.p_value, "n": d.n, "significant": d.significant}


def _fmt(e: dict[str, Any] | None, digits: int = 2) -> str:
    if not e or e.get("n", 0) == 0:
        return "pending"
    return f"{e['mean']:.{digits}f} [{e['low']:.{digits}f}, {e['high']:.{digits}f}]"


def _fmt_diff(d: dict[str, Any] | None) -> str:
    if not d:
        return "pending"
    star = " *" if d["significant"] else ""
    return f"{d['diff']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]{star}"


def build(configs: Sequence[RunConfig] = MAIN_CONFIGS, best: RunConfig = BEST_CONFIG) -> dict[str, Any]:
    rows = [s for cfg in configs if (s := config_summary(APP_MODEL, cfg))]
    reference = config_summary(REF_MODEL, best)
    vs_best = {
        cfg.name: {m: paired(APP_MODEL, cfg, APP_MODEL, best, m) for m in METRICS} for cfg in configs if cfg != best
    }
    model_diff = {m: paired(APP_MODEL, best, REF_MODEL, best, m) for m in (*METRICS, "citation_coverage")}
    agreement_path = RESULTS / "judge_agreement.json"
    agreement = json.loads(agreement_path.read_text(encoding="utf-8")) if agreement_path.exists() else None
    judge_meta = next(
        (json.loads(p.read_text(encoding="utf-8")) for p in sorted((RESULTS / "judge").glob("*/scores_*.meta.json"))),
        None,
    )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "app_model": APP_MODEL,
        "reference_model": REF_MODEL,
        "best_config": best.name,
        "configs": rows,
        "reference": reference,
        "paired_vs_best": vs_best,
        "app_vs_reference": model_diff,
        "judge": judge_meta["judge"] if judge_meta else None,
        "judge_agreement": agreement,
        "retrieval_metrics_file": "results/retrieval_metrics.csv",
    }
    write_json(SUMMARY, summary)
    _write_tables(rows, reference, vs_best, model_diff, best)
    _figures(rows, reference)
    return summary


def _write_tables(
    rows: list[dict[str, Any]],
    reference: dict[str, Any] | None,
    vs_best: dict[str, dict[str, Any]],
    model_diff: dict[str, Any],
    best: RunConfig,
) -> None:
    lines = [
        "## Generation and RAGAS results",
        "",
        "Means with 95% bootstrap CIs over questions. Δ = paired bootstrap difference; * = CI excludes 0.",
        "",
        f"### {APP_MODEL} on the evaluated configurations",
        "",
        "| Chunking | Retrieval | Faithfulness | Answer relevancy | Context precision | Context recall | "
        "Refusal (out-of-corpus) | False refusal | Cited sentences |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['chunking']} | {r['retrieval']} | "
            + " | ".join(_fmt(r[m]) for m in METRICS)
            + f" | {_fmt(r['refusal_rate'])} | {_fmt(r['false_refusal_rate'])} | {_fmt(r['citation_coverage'])} |"
        )
    lines += [
        "",
        f"### Paired differences vs `{best.name}` ({APP_MODEL})",
        "",
        "| Configuration | " + " | ".join(METRICS) + " |",
        "|---|" + "---|" * len(METRICS),
    ]
    for name, diffs in vs_best.items():
        lines.append(f"| {name} | " + " | ".join(_fmt_diff(diffs[m]) for m in METRICS) + " |")
    if reference:
        app = next((r for r in rows if r["config"] == best.name), None)
        lines += [
            "",
            f"### {APP_MODEL} vs {REF_MODEL} on `{best.name}` (same contexts, same judge)",
            "",
            f"| Metric | {APP_MODEL} | {REF_MODEL} | Δ (app - reference) |",
            "|---|---|---|---|",
        ]
        for m in (*METRICS, "citation_coverage"):
            lines.append(
                f"| {m} | {_fmt(app[m] if app else None)} | {_fmt(reference[m])} | {_fmt_diff(model_diff[m])} |"
            )
        lines.append("")
        lines.append("Context precision and recall depend only on retrieval: both generators share them.")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    flat = []
    for r in [*rows, *([reference] if reference else [])]:
        row = {
            k: r[k]
            for k in ("model", "config", "n_factual", "n_impossible", "tokens_mean", "seconds_mean", "generated_at")
        }
        for m in (*METRICS, "refusal_rate", "false_refusal_rate", "citation_coverage"):
            e = r[m] or {}
            row.update({m: e.get("mean"), f"{m}_low": e.get("low"), f"{m}_high": e.get("high"), f"{m}_n": e.get("n")})
        flat.append(row)
    if flat:
        with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)


def _figures(rows: list[dict[str, Any]], reference: dict[str, Any] | None) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = []
    judged = [r for r in rows if r["judged"]]
    if judged:
        fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 0.5 * len(judged) + 1.6), sharey=True)
        labels = [f"{r['chunking']} + {r['retrieval']}" for r in judged]
        for ax, metric in zip(axes, METRICS, strict=True):
            for y, r in enumerate(judged):
                e = r[metric]
                if e and e["n"]:
                    ax.errorbar(
                        e["mean"],
                        y,
                        xerr=[[e["mean"] - e["low"]], [e["high"] - e["mean"]]],
                        fmt="o",
                        color="#2b6cb0",
                        capsize=3,
                    )
            ax.set_title(metric.replace("_", " "))
            ax.set_xlim(0, 1.05)
            ax.grid(axis="x", alpha=0.3)
        axes[0].set_yticks(range(len(judged)), labels)
        axes[0].invert_yaxis()
        fig.suptitle(f"{APP_MODEL}: RAGAS metrics by configuration (mean, 95% bootstrap CI)")
        fig.tight_layout()
        path = FIGURES / "ragas_by_config.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path)
    app = next((r for r in rows if r["config"] == BEST_CONFIG.name and r["judged"]), None)
    if app and reference and reference["judged"]:
        metrics = ["faithfulness", "answer_relevancy", "citation_coverage"]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for k, (r, colour) in enumerate(((app, "#2b6cb0"), (reference, "#dd6b20"))):
            xs = [i + (k - 0.5) * 0.3 for i in range(len(metrics))]
            means = [r[m]["mean"] for m in metrics]
            errs = [[r[m]["mean"] - r[m]["low"] for m in metrics], [r[m]["high"] - r[m]["mean"] for m in metrics]]
            ax.bar(xs, means, width=0.3, yerr=errs, capsize=3, color=colour, label=r["model"])
        ax.set_xticks(range(len(metrics)), [m.replace("_", " ") for m in metrics])
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
        ax.set_title(f"{BEST_CONFIG.name}: application vs reference generator (95% CI)")
        fig.tight_layout()
        path = FIGURES / "app_vs_reference.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path)
    return outputs
