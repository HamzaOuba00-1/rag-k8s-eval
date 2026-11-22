"""Two-phase, resumable evaluation runs.

Phase 1 - retrieve: for one configuration (strategy, method), retrieve the
context (ranked chunks cut to the same character budget for every strategy)
for every evaluation question and store them in
results/retrieval/<strategy>_<method>.jsonl. Deterministic and fast; it is the
input of the retrieval metrics and of phase 2.

Phase 2 - generate: answer every question from the stored contexts with Ollama.
Each answer is cached in .cache/generations/<key>.json, where the key hashes
everything that determines the answer (model and digest, prompt version,
decoding options, question, contexts). An interrupted run resumes where it
stopped; identical (question, contexts) pairs are never generated twice, even
across configurations. The run is then assembled into
results/runs/run_<strategy>_<method>.jsonl plus a .meta.json file.

The phases run separately so that the embedding and reranking models are not
in memory while the LLM runs.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__, config, generation
from .citations import check_citations
from .config import Method, Strategy
from .refusal import is_refusal
from .retrieval import Hit
from .utils import read_jsonl, sha256_file, sha256_text, write_json, write_jsonl

QUESTIONS_PATH = config.EVAL_DIR / "questions.jsonl"
RETRIEVAL_DIR = config.RESULTS_DIR / "retrieval"
RUNS_DIR = config.RESULTS_DIR / "runs"
GENERATION_CACHE = config.CACHE_DIR / "generations"


@dataclass(frozen=True)
class RunConfig:
    strategy: Strategy
    method: Method

    @property
    def name(self) -> str:
        return f"{self.strategy}_{self.method}"

    @classmethod
    def parse(cls, text: str) -> RunConfig:
        """``"structure:hybrid_rerank"`` -> RunConfig."""
        strategy, _, method = text.partition(":")
        if strategy not in config.STRATEGIES or method not in config.METHODS:
            raise ValueError(f"expected <strategy>:<method>, got {text!r}")
        return cls(strategy, method)


ALL_CONFIGS: tuple[RunConfig, ...] = tuple(RunConfig(s, m) for s in config.STRATEGIES for m in config.METHODS)


# --- Questions -------------------------------------------------------------------


def question_id(question: str) -> str:
    """Stable id derived from the question text (independent of file order)."""
    return "q_" + sha256_text(question.strip())[:10]


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    questions = read_jsonl(path)
    for q in questions:
        q.setdefault("category", "factual")
        q["id"] = question_id(q["question"])
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate questions")
    return questions


# --- Phase 1: retrieval -----------------------------------------------------------


def retrieval_path(cfg: RunConfig) -> Path:
    return RETRIEVAL_DIR / f"{cfg.name}.jsonl"


def _hit_record(hit: Hit) -> dict[str, Any]:
    c = hit.chunk
    return {
        "rank": hit.rank,
        "score": hit.score,
        "rerank_score": hit.rerank_score,
        "chunk_id": c["chunk_id"],
        "doc_id": c["doc_id"],
        "title": c["title"],
        "source_url": c["source_url"],
        "text": c["text"],
    }


def hits_from_records(records: Sequence[dict[str, Any]]) -> list[Hit]:
    return [Hit(chunk=r, score=r["score"], rank=r["rank"], rerank_score=r.get("rerank_score")) for r in records]


def retrieve(
    configs: Sequence[RunConfig],
    budget_chars: int = config.CONTEXT_BUDGET_CHARS,
    questions_path: Path = QUESTIONS_PATH,
) -> list[Path]:
    """Phase 1. Strategies are loaded once; models are shared across configurations."""
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from tqdm import tqdm

    from .retrieval import Retriever

    questions = load_questions(questions_path)
    embed = SentenceTransformer(config.EMBEDDING_MODEL)
    rerank = CrossEncoder(config.RERANKER_MODEL) if any(c.method == "hybrid_rerank" for c in configs) else None
    outputs = []
    for strategy in dict.fromkeys(c.strategy for c in configs):
        retriever = Retriever(strategy, embed_model=embed, cross_encoder=rerank)
        index_sha = json.loads((config.INDEX_DIR / strategy / "index_meta.json").read_text(encoding="utf-8"))[
            "chunks_sha256"
        ]
        for cfg in (c for c in configs if c.strategy == strategy):
            records = []
            for q in tqdm(questions, desc=f"retrieve {cfg.name}", leave=False):
                hits = retriever.search(cfg.method, q["question"], budget_chars=budget_chars)
                records.append(
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "category": q["category"],
                        "ground_truth": q.get("ground_truth"),
                        "source_doc_id": q.get("source_doc_id"),
                        "origin": q.get("origin", "structure_chunk") if q["category"] == "factual" else None,
                        "hits": [_hit_record(h) for h in hits],
                    }
                )
            out = retrieval_path(cfg)
            write_jsonl(out, records)
            write_json(
                out.with_suffix(".meta.json"),
                {
                    "config": cfg.name,
                    "budget_chars": budget_chars,
                    "max_context_chunks": config.MAX_CONTEXT_CHUNKS,
                    "n_questions": len(records),
                    "questions_sha256": sha256_file(questions_path),
                    "chunks_sha256": index_sha,
                    "embedding_model": config.EMBEDDING_MODEL,
                    "reranker_model": config.RERANKER_MODEL if cfg.method == "hybrid_rerank" else None,
                    "created_at": _now(),
                },
            )
            print(f"{cfg.name:<26} {len(records)} questions -> {out}")
            outputs.append(out)
    return outputs


# --- Phase 2: generation ----------------------------------------------------------


def generation_key(question: str, contexts: Sequence[str], backend: generation.Backend) -> str:
    payload = {
        "provider": backend.provider,
        "model": backend.model,
        "digest": backend.digest,
        "params": backend.params,
        "prompt": generation.PROMPT_VERSION,
        "system": generation.SYSTEM_PROMPT,
        "question": question,
        "contexts": list(contexts),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


# Selected on 2026-11-22 by the rule fixed before any LLM evaluation (best retrieval MRR,
# results/retrieval_metrics.md): the 3 chunkings with the best retrieval method, and the
# 4 retrieval methods with the best chunking.
MAIN_CONFIGS: tuple[RunConfig, ...] = tuple(
    RunConfig.parse(c)
    for c in (
        "fixed:hybrid_rerank",
        "structure:hybrid_rerank",
        "semantic:hybrid_rerank",
        "fixed:bm25",
        "fixed:dense",
        "fixed:hybrid",
    )
)
BEST_CONFIG = MAIN_CONFIGS[0]


def _generate_with_retry(
    backend: generation.Backend, question: str, hits: Sequence[Hit], retries: int = 2
) -> generation.Generation:
    from .groq_client import DailyQuotaReached, GroqError

    for attempt in range(retries + 1):
        try:
            return backend.generate(question, hits)
        except (DailyQuotaReached, GroqError):
            raise  # the Groq client already retries what can be retried
        except Exception as exc:  # ollama raises ResponseError / ConnectionError / httpx errors
            if attempt == retries:
                raise RuntimeError(
                    f"Generation failed after {retries + 1} attempts ({type(exc).__name__}: {exc}). "
                    "Check the backend, then rerun the same command: finished answers are cached."
                ) from exc
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def run_path(cfg: RunConfig, backend: generation.Backend) -> Path:
    return RUNS_DIR / backend.slug / f"run_{cfg.name}.jsonl"


def generate_run(
    cfg: RunConfig,
    backend: generation.Backend,
    cache_dir: Path = GENERATION_CACHE,
    limit: int | None = None,
) -> Path:
    """Phase 2 for one configuration. Safe to interrupt and rerun (answers are cached one by one)."""
    from tqdm import tqdm

    src = retrieval_path(cfg)
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found: run `python -m ragk8s retrieve --configs {cfg.strategy}:{cfg.method}` first"
        )
    items = read_jsonl(src)[:limit] if limit else read_jsonl(src)

    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    n_cached, new_seconds = 0, 0.0
    bar = tqdm(items, desc=f"{backend.slug} {cfg.name}")
    for item in bar:
        hits = hits_from_records(item["hits"])
        contexts = [h.chunk["text"] for h in hits]
        cache_file = cache_dir / f"{generation_key(item['question'], contexts, backend)}.json"
        if cache_file.exists():
            gen = json.loads(cache_file.read_text(encoding="utf-8"))
            n_cached += 1
        else:
            gen = _generate_with_retry(backend, item["question"], hits).as_dict()
            write_json(cache_file, gen)
            new_seconds += gen.get("total_seconds") or 0.0
        bar.set_postfix(cached=n_cached, new=len(records) + 1 - n_cached)
        check = check_citations(gen["answer"], len(hits))
        records.append(
            {
                "id": item["id"],
                "question": item["question"],
                "category": item["category"],
                "origin": item.get("origin"),
                "ground_truth": item["ground_truth"],
                "source_doc_id": item["source_doc_id"],
                "answer": gen["answer"],
                "is_refusal": is_refusal(gen["answer"]),
                "cited": list(check.cited),
                "invalid_citations": list(check.invalid),
                "citation_coverage": check.coverage,
                "contexts": contexts,
                "sources": [
                    {"n": i, **{k: h.chunk[k] for k in ("chunk_id", "doc_id", "title", "source_url")}}
                    for i, h in enumerate(hits, 1)
                ],
                "generation": {k: gen.get(k) for k in ("model", "prompt_tokens", "completion_tokens", "total_seconds")},
            }
        )

    out = run_path(cfg, backend)
    write_jsonl(out, records)
    seconds = [r["generation"]["total_seconds"] for r in records if r["generation"]["total_seconds"]]
    tokens = [(r["generation"]["prompt_tokens"] or 0) + (r["generation"]["completion_tokens"] or 0) for r in records]
    write_json(
        out.with_suffix(".meta.json"),
        {
            "config": cfg.name,
            "n_questions": len(records),
            "complete": limit is None,
            "provider": backend.provider,
            "model": backend.model,
            "models_returned": sorted({str(r["generation"]["model"]) for r in records if r["generation"]["model"]}),
            "model_digest": backend.digest,
            "ollama_version": _ollama_version() if backend.provider == "ollama" else None,
            "prompt_version": generation.PROMPT_VERSION,
            "params": backend.params,
            "retrieval_meta": json.loads(src.with_suffix(".meta.json").read_text(encoding="utf-8")),
            "tokens_total": sum(tokens),
            "tokens_mean": round(sum(tokens) / len(tokens), 1) if tokens else None,
            "generation_seconds_total": round(sum(seconds), 1),
            "generation_seconds_mean": round(sum(seconds) / len(seconds), 2) if seconds else None,
            "new_generation_seconds_this_invocation": round(new_seconds, 1),
            "ragk8s_version": __version__,
            "git_commit": _git_commit(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "created_at": _now(),
        },
    )
    print(f"{cfg.name:<26} {len(records)} answers ({n_cached} from cache) -> {out}")
    return out


# --- Helpers ----------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git_commit() -> str | None:
    """HEAD of the working copy; on Kaggle (no .git) the commit recorded by the notebook."""
    if os.environ.get("RAGK8S_GIT_COMMIT"):
        return os.environ["RAGK8S_GIT_COMMIT"]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("-dirty" if dirty else "")


def _ollama_version() -> str | None:
    import urllib.request

    try:
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        host = host if host.startswith("http") else f"http://{host}"
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/version", timeout=5) as resp:
            return str(json.load(resp).get("version"))
    except Exception:
        return None
