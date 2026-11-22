"""Command-line interface: ``python -m ragk8s <command>``.

Pipeline, in order:
  ingest   fetch the docs at the pinned commit, or verify the frozen snapshot
  chunk    build data/processed/chunks_<strategy>.jsonl
  index    build BM25 + FAISS indexes in data/index/<strategy>/
  search   inspect retrieval results for one query (no LLM)
  ask      answer one question with citations (needs Ollama)
  retrieve phase 1: store the top-k contexts of every evaluation question (no LLM)
  generate phase 2: answer from the stored contexts (Ollama; cached and resumable)
  run      retrieve, then generate
  retrieval-metrics  hit@k and MRR of the source document per configuration (no LLM)
  questions draft  extra gold questions from whole pages (Groq, neutral w.r.t. chunking)
  review   export the gold questions to Excel for human review, then apply the verdicts
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from . import config


def _cmd_ingest(args: argparse.Namespace) -> int:
    from . import ingestion

    if args.verify:
        problems = ingestion.verify()
        for p in problems:
            print(p)
        print("Snapshot OK." if not problems else f"{len(problems)} problem(s).")
        return 0 if not problems else 1

    clone = ingestion.fetch_corpus()
    docs = ingestion.parse_corpus(clone)
    snapshot_exists = (config.RAW_DIR / "manifest.json").exists()
    if snapshot_exists and not args.force:
        problems = ingestion.compare_with_snapshot(docs)
        for p in problems[:20]:
            print(p)
        status = "identical to" if not problems else f"{len(problems)} difference(s) with"
        print(f"{len(docs)} documents parsed, {status} the frozen snapshot. Use --force to overwrite.")
        rc = 0 if not problems else 1
    else:
        ingestion.export(docs)
        print(f"{len(docs)} documents written to {config.RAW_DIR}")
        rc = 0
    if not args.keep_clone:
        ingestion.remove_clone()
    return rc


def _cmd_chunk(args: argparse.Namespace) -> int:
    from . import chunking, ingestion
    from .utils import set_seed

    set_seed(config.SEED)
    chunking.run(ingestion.load_raw_docs(), args.strategies)
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    from . import indexing
    from .utils import set_seed

    set_seed(config.SEED)
    for strategy in args.strategies:
        indexing.build(strategy)
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from .retrieval import Retriever

    retriever = Retriever(args.strategy)
    for method in args.methods:
        print(f"\n### {method} ###")
        for hit in retriever.search(method, args.query, budget_chars=args.budget):
            score = hit.rerank_score if hit.rerank_score is not None else hit.score
            preview = hit.chunk["text"][:120].replace("\n", " ")
            print(f"  [{hit.rank}] {score:.4f} | {hit.chunk['title']} ({hit.chunk['doc_id']})\n      {preview}...")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from . import generation
    from .citations import check_citations
    from .refusal import is_refusal
    from .retrieval import Retriever

    hits = Retriever(args.strategy).search(args.method, args.question, budget_chars=args.budget)
    result = generation.make_backend(args.provider, args.model).generate(args.question, hits)
    print(result.answer)
    check = check_citations(result.answer, len(hits))
    print(
        f"\n[refusal={is_refusal(result.answer)} cited={list(check.cited)} invalid={list(check.invalid)} "
        f"coverage={check.coverage:.0%} tokens={result.prompt_tokens}+{result.completion_tokens} "
        f"time={result.total_seconds or 0:.1f}s]"
    )
    print("\nSources:")
    for s in generation.sources(hits):
        print(f"  [{s['n']}] {s['title']} - {s['url']}")
    return 0


def _configs(args: argparse.Namespace) -> list[Any]:
    from .pipeline import ALL_CONFIGS, BEST_CONFIG, MAIN_CONFIGS, RunConfig

    if args.all:
        return list(ALL_CONFIGS)
    if getattr(args, "preset", None) == "main":
        return list(MAIN_CONFIGS)
    if getattr(args, "preset", None) == "best":
        return [BEST_CONFIG]
    if not args.configs:
        raise SystemExit("give --configs <strategy>:<method> ..., --preset main|best or --all")
    return [RunConfig.parse(c) for c in args.configs]


def _cmd_retrieve(args: argparse.Namespace) -> int:
    from . import pipeline

    pipeline.retrieve(_configs(args), budget_chars=args.budget)
    return 0


def _generate_all(args: argparse.Namespace) -> None:
    from . import generation, pipeline

    backend = generation.make_backend(args.provider, args.model)
    try:
        for cfg in _configs(args):
            pipeline.generate_run(cfg, backend, limit=args.limit)
    finally:
        client = backend.extra_meta.get("client")
        if client is not None:
            tokens, requests = client.ledger.used(backend.model)
            print(f"Groq usage today (UTC) for {backend.model}: {tokens:,} tokens, {requests} requests")


def _cmd_generate(args: argparse.Namespace) -> int:
    _generate_all(args)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Both phases; retrieval models are released before the LLM runs."""
    import gc

    from . import pipeline

    pipeline.retrieve(_configs(args), budget_chars=args.budget)
    gc.collect()
    _generate_all(args)
    return 0


def _cmd_retrieval_metrics(args: argparse.Namespace) -> int:
    from . import retrieval_metrics
    from .pipeline import RunConfig

    configs = None if (args.all or not args.configs) else _configs(args)
    reference = RunConfig.parse(args.reference) if args.reference else None
    retrieval_metrics.run(configs, reference)
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    from . import review

    if args.action == "export":
        print(f"Review file -> {review.export(overwrite=args.overwrite)}")
    else:
        counts = review.apply(reviewer=args.reviewer, assisted_by=args.assisted_by)
        print(
            f"Applied: {counts['keep']} kept, {counts['fix']} fixed, {counts['drop']} dropped"
            f" -> {review.QUESTIONS_PATH}"
        )
    return 0


def _cmd_questions(args: argparse.Namespace) -> int:
    from . import questions_draft

    stats = questions_draft.draft(n=args.n, model=args.model, replace=args.replace)
    print(
        f"{stats['pages']} pages sampled: {stats['accepted']} drafts accepted, {stats['rejected']} rejected "
        f"(see {questions_draft.REJECTED_PATH}); {stats['new_tokens']:,} new Groq tokens."
    )
    return 0


def _cmd_kaggle(args: argparse.Namespace) -> int:
    from pathlib import Path

    from . import kaggle

    if args.action == "export":
        configs = _configs(args)
        name = args.name or kaggle.default_name(configs)
        out = kaggle.export(name, configs, run_models=args.runs or (), include_judge_cache=args.with_judge_cache)
        print(f"Kaggle dataset archive -> {out}")
    else:
        if not args.path:
            raise SystemExit("give the path of the Kaggle output (zip or folder)")
        stats = kaggle.import_outputs(Path(args.path))
        print(f"Imported {stats['copied']} files ({stats['identical']} already identical): {', '.join(stats['runs'])}")
        for conflict in stats["conflicts"]:
            print(f"  CONFLICT, not overwritten: {conflict}")
    return 0


def _cmd_judge(args: argparse.Namespace) -> int:
    from . import judge
    from .pipeline import RUNS_DIR

    wanted = [RUNS_DIR / model / f"run_{cfg.name}.jsonl" for model in args.models for cfg in _configs(args)]
    files = [f for f in wanted if f.exists()]
    for f in wanted:
        if not f.exists():
            print(f"skipped (no run yet): {f}")
    if not files:
        raise SystemExit("no run to judge")
    stats = judge.run(
        files,
        max_minutes=args.max_minutes,
        concurrency=args.concurrency,
        limit=args.limit,
        retry_errors=args.retry_errors,
    )
    print(json.dumps({k: v for k, v in stats.items() if k != "outputs"}))
    return 0


def _cmd_validation(args: argparse.Namespace) -> int:
    from . import validation

    if args.action == "export":
        print(f"Judge validation sheet -> {validation.export(n=args.n, overwrite=args.overwrite)}")
    else:
        result = validation.agree()
        for metric, a in result["metrics"].items():
            print(
                f"{metric:<18} n={a['n']} spearman={a['spearman']:.2f}"
                f" [{a['spearman_ci'][0]:.2f}, {a['spearman_ci'][1]:.2f}]"
                f" mae={a['mae']:.2f} bias={a['bias_judge_minus_human']:+.2f} kappa={a['kappa_binary_0.5']:.2f}"
            )
        print(f"-> {validation.OUT_PATH}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from . import compare

    compare.build()
    print(compare.OUT_MD.read_text(encoding="utf-8"))
    print(f"-> {compare.OUT_MD}, {compare.OUT_CSV}, {compare.SUMMARY}, {compare.FIGURES}/")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from . import report

    for path, blocks in report.run().items():
        print(f"{path}: {', '.join(blocks) or 'no result block'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragk8s", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="fetch the corpus at the pinned commit, or verify the snapshot")
    p.add_argument("--verify", action="store_true", help="only check data/raw against the manifest hashes (offline)")
    p.add_argument("--force", action="store_true", help="overwrite data/raw with the fresh ingestion")
    p.add_argument("--keep-clone", action="store_true", help="keep the temporary upstream checkout")
    p.set_defaults(func=_cmd_ingest)

    for name, func, help_ in (("chunk", _cmd_chunk, "build chunk files"), ("index", _cmd_index, "build indexes")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--strategies", nargs="+", choices=config.STRATEGIES, default=list(config.STRATEGIES))
        p.set_defaults(func=func)

    p = sub.add_parser("search", help="show retrieval results for a query")
    p.add_argument("query")
    p.add_argument("--strategy", choices=config.STRATEGIES, default="structure")
    p.add_argument("--methods", nargs="+", choices=config.METHODS, default=list(config.METHODS))
    p.add_argument("--budget", type=int, default=config.CONTEXT_BUDGET_CHARS, help="context budget in characters")
    p.set_defaults(func=_cmd_search)

    p = sub.add_parser("ask", help="answer one question with citations (needs Ollama)")
    p.add_argument("question")
    p.add_argument("--provider", choices=("ollama", "groq"), default="ollama")
    p.add_argument("--model", default=None)
    p.add_argument("--strategy", choices=config.STRATEGIES, default="structure")
    p.add_argument("--method", choices=config.METHODS, default="hybrid_rerank")
    p.add_argument("--budget", type=int, default=config.CONTEXT_BUDGET_CHARS, help="context budget in characters")
    p.set_defaults(func=_cmd_ask)

    for name, func, help_ in (
        ("retrieve", _cmd_retrieve, "phase 1: retrieve contexts for all evaluation questions (no LLM)"),
        ("generate", _cmd_generate, "phase 2: answer from stored contexts (Ollama, cached, resumable)"),
        ("run", _cmd_run, "retrieve then generate"),
    ):
        p = sub.add_parser(name, help=help_)
        group = p.add_mutually_exclusive_group()
        group.add_argument("--configs", nargs="+", metavar="STRATEGY:METHOD", help="e.g. structure:hybrid_rerank")
        group.add_argument("--all", action="store_true", help="all 12 strategy x method combinations")
        group.add_argument("--preset", choices=("main", "best"), help="main = the 6 evaluated configurations")
        if name != "retrieve":
            p.add_argument("--provider", choices=("ollama", "groq"), default="ollama")
            p.add_argument(
                "--model", default=None, help="default: qwen2.5:7b-instruct (ollama), qwen/qwen3.8-27b (groq)"
            )
        p.add_argument("--budget", type=int, default=config.CONTEXT_BUDGET_CHARS, help="context budget in characters")
        p.add_argument("--limit", type=int, default=None, help="generate only the first N questions (smoke test)")
        p.set_defaults(func=func)

    p = sub.add_parser("retrieval-metrics", help="hit@k / MRR of the source document, with bootstrap CIs (no LLM)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--configs", nargs="+", metavar="STRATEGY:METHOD")
    group.add_argument("--all", action="store_true", help="all configurations with a retrieval file (default)")
    p.add_argument("--reference", metavar="STRATEGY:METHOD", help="paired MRR differences vs this configuration")
    p.set_defaults(func=_cmd_retrieval_metrics)

    p = sub.add_parser("questions", help="draft extra gold questions from whole pages (Groq)")
    p.add_argument("action", choices=("draft",))
    p.add_argument("--n", type=int, default=25, help="number of pages to sample")
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--replace", action="store_true", help="also discard already reviewed page questions")
    p.set_defaults(func=_cmd_questions)

    p = sub.add_parser("judge", help="RAGAS scores of generation runs with a local judge (Kaggle GPU)")
    p.add_argument(
        "--models", nargs="+", required=True, metavar="MODEL_DIR", help="e.g. qwen3.8-27b qwen2.5-7b-instruct"
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--configs", nargs="+", metavar="STRATEGY:METHOD")
    group.add_argument("--all", action="store_true")
    group.add_argument("--preset", choices=("main", "best"))
    p.add_argument("--max-minutes", type=float, default=None, help="stop starting new scores after this time")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="first N records of each run (pilot)")
    p.add_argument("--retry-errors", action="store_true", help="score again the cached judge errors")
    p.set_defaults(func=_cmd_judge)

    p = sub.add_parser("validation", help="human validation of the judge (Excel sheet, then agreement)")
    p.add_argument("action", choices=("export", "agree"))
    p.add_argument("--n", type=int, default=40, help="export: number of answers to label")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_cmd_validation)

    p = sub.add_parser("compare", help="final tables, figures and results/summary.json")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("report", help="regenerate the result blocks of README.md and REPORT.md")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("kaggle", help="export contexts for a Kaggle notebook / import its outputs")
    p.add_argument("action", choices=("export", "import"))
    p.add_argument("path", nargs="?", help="import: Kaggle output zip or folder")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--configs", nargs="+", metavar="STRATEGY:METHOD")
    group.add_argument("--all", action="store_true")
    group.add_argument("--preset", choices=("main", "best"))
    p.add_argument("--name", default=None, help="export: archive name (default from the configurations)")
    p.add_argument("--runs", nargs="+", metavar="MODEL_DIR", help="export: also the complete runs of these models")
    p.add_argument("--with-judge-cache", action="store_true", help="export: include .cache/judge to resume judging")
    p.set_defaults(func=_cmd_kaggle)

    p = sub.add_parser("review", help="human review of the gold questions (Excel round trip)")
    p.add_argument("action", choices=("export", "apply"))
    p.add_argument("--overwrite", action="store_true", help="replace an existing review file on export")
    p.add_argument("--reviewer", default="human", help="who reviewed and validated the verdicts (reviewed_by)")
    p.add_argument("--assisted-by", default=None, help="tool that proposed the verdicts, if any (review_assisted_by)")
    p.set_defaults(func=_cmd_review)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from .groq_client import DailyQuotaReached, GroqError

    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except DailyQuotaReached as exc:
        print(f"Daily quota reached: {exc}", file=sys.stderr)
        return 3
    except GroqError as exc:
        print(f"Groq error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
