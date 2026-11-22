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
"""

from __future__ import annotations

import argparse
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
        for hit in retriever.search(method, args.query, top_k=args.top_k):
            score = hit.rerank_score if hit.rerank_score is not None else hit.score
            preview = hit.chunk["text"][:120].replace("\n", " ")
            print(f"  [{hit.rank}] {score:.4f} | {hit.chunk['title']} ({hit.chunk['doc_id']})\n      {preview}...")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from . import generation
    from .citations import check_citations
    from .refusal import is_refusal
    from .retrieval import Retriever

    hits = Retriever(args.strategy).search(args.method, args.question, top_k=args.top_k)
    result = generation.generate(args.question, hits)
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
    from .pipeline import ALL_CONFIGS, RunConfig

    if args.all:
        return list(ALL_CONFIGS)
    if not args.configs:
        raise SystemExit("give --configs <strategy>:<method> ... or --all")
    return [RunConfig.parse(c) for c in args.configs]


def _cmd_retrieve(args: argparse.Namespace) -> int:
    from . import pipeline

    pipeline.retrieve(_configs(args), top_k=args.top_k)
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from . import pipeline

    for cfg in _configs(args):
        pipeline.generate_run(cfg, limit=args.limit)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Both phases; retrieval models are released before the LLM runs."""
    import gc

    from . import pipeline

    configs = _configs(args)
    pipeline.retrieve(configs, top_k=args.top_k)
    gc.collect()
    for cfg in configs:
        pipeline.generate_run(cfg, limit=args.limit)
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
    p.add_argument("--top-k", type=int, default=config.TOP_K)
    p.set_defaults(func=_cmd_search)

    p = sub.add_parser("ask", help="answer one question with citations (needs Ollama)")
    p.add_argument("question")
    p.add_argument("--strategy", choices=config.STRATEGIES, default="structure")
    p.add_argument("--method", choices=config.METHODS, default="hybrid_rerank")
    p.add_argument("--top-k", type=int, default=config.TOP_K)
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
        p.add_argument("--top-k", type=int, default=config.TOP_K)
        p.add_argument("--limit", type=int, default=None, help="generate only the first N questions (smoke test)")
        p.set_defaults(func=func)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
