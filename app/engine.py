"""What the API does: live retrieval with the ONNX models, live or precomputed answers.

Live answers use LIVE_CONFIG (fixed chunking, hybrid BM25 + dense). The evaluated best
configuration adds the cross-encoder reranker, which takes seconds per query on one
vCPU (results/onnx_parity_*.json), so its results are served precomputed for the
evaluation questions, from the evaluation runs themselves.
"""

from __future__ import annotations

import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ragk8s.citations import check_citations
from ragk8s.generation import Generation
from ragk8s.refusal import is_refusal
from ragk8s.retrieval import Hit

from .quota import QuotaGate, VisitorLimiter

LIVE_STRATEGY = "fixed"
LIVE_METHOD = "hybrid"
LIVE_METHODS = ("bm25", "dense", "hybrid")

DEMO_HINT = "The sources below were retrieved live. Example questions still show full answers."


class RetrieverLike(Protocol):
    def search(self, method: Any, query: str, budget_chars: int = ...) -> list[Hit]: ...


class Busy(RuntimeError):
    """Groq asked to wait longer than a web request can."""


class NoReranker:
    """Placeholder cross-encoder: the app never reranks live."""

    def predict(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("the reranker is not loaded in the web app; hybrid_rerank is served precomputed")


class LazyRetrievers(Mapping[str, RetrieverLike]):
    """Retrievers by strategy, each built on first use (once, under a lock).

    Live answers need only LIVE_STRATEGY; the other strategies serve the Compare page,
    so loading them at startup would only lengthen Cloud Run cold starts.
    """

    def __init__(self, strategies: Sequence[str], build: Callable[[str], RetrieverLike]) -> None:
        self._strategies = tuple(strategies)
        self._build = build
        self._built: dict[str, RetrieverLike] = {}
        self._lock = threading.Lock()

    def __getitem__(self, strategy: str) -> RetrieverLike:
        if strategy not in self._strategies:
            raise KeyError(strategy)
        if strategy not in self._built:
            with self._lock:
                if strategy not in self._built:
                    self._built[strategy] = self._build(strategy)
        return self._built[strategy]

    def __iter__(self) -> Iterator[str]:
        return iter(self._strategies)

    def __len__(self) -> int:
        return len(self._strategies)

    @property
    def loaded(self) -> list[str]:
        return list(self._built)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip("?").strip().casefold()


def _source(n: int, hit: Hit) -> dict[str, Any]:
    c = hit.chunk
    return {
        "n": n,
        "title": c["title"],
        "url": c["source_url"],
        "doc_id": c["doc_id"],
        "chunk_id": c["chunk_id"],
        "text": c["text"],
        "score": float(hit.rerank_score if hit.rerank_score is not None else hit.score),
    }


def _source_rank(sources: Sequence[Mapping[str, Any]], doc_id: str | None) -> int | None:
    if not doc_id:
        return None
    return next((int(s["n"]) for s in sources if s["doc_id"] == doc_id), None)


LiveLLM = Callable[[str, Sequence[Hit]], Generation]


def groq_llm(
    api_key: str, model: str | None = None, max_wait: float = 3.0, timeout: float = 30.0
) -> tuple[LiveLLM, str]:
    """Groq backend for web requests: never sleeps more than ``max_wait`` seconds and
    never waits more than ``timeout`` seconds for an answer (the evaluation allows 120 s)."""
    import httpx

    from ragk8s import config
    from ragk8s.generation import groq_backend
    from ragk8s.groq_client import BASE_URL, GroqClient, UsageLedger

    def short_sleep(seconds: float) -> None:
        if seconds > max_wait:
            raise Busy(f"Groq asked to wait {seconds:.0f} s")
        time.sleep(seconds)

    ledger = UsageLedger(Path(tempfile.gettempdir()) / "doctruth_groq_usage.json")
    http = httpx.Client(base_url=BASE_URL, timeout=timeout)
    client = GroqClient(api_key=api_key, ledger=ledger, sleep=short_sleep, max_retries=1, http=http)
    backend = groq_backend(model or config.GROQ_MODEL, client=client)
    return backend.generate, backend.model


class Engine:
    def __init__(
        self,
        bundle: Mapping[str, Any],
        retrievers: Mapping[str, RetrieverLike],
        gate: QuotaGate,
        llm: LiveLLM | None = None,
        llm_model: str = "",
        search_limiter: VisitorLimiter | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.bundle = bundle
        self.retrievers = retrievers
        self.gate = gate
        self.llm = llm
        self.llm_model = llm_model
        self.search_limiter = search_limiter or VisitorLimiter(120)
        self._clock = clock
        self.questions = {q["id"]: q for q in bundle.get("questions", [])}
        self._by_text = {normalise(q["question"]): qid for qid, q in self.questions.items()}
        self.budget_chars = int(bundle.get("budget_chars", 5000))

    # --- helpers ---------------------------------------------------------------------

    def evaluation_question(self, text: str) -> dict[str, Any] | None:
        qid = self._by_text.get(normalise(text))
        return self.questions[qid] if qid else None

    def examples(self) -> list[dict[str, Any]]:
        return [
            {"id": q["id"], "question": q["question"], "category": q["category"], "source_url": q.get("source_url")}
            for q in self.questions.values()
        ]

    # --- /api/ask --------------------------------------------------------------------

    def ask(self, question: str, ip: str) -> dict[str, Any]:
        question = question.strip()
        q = self.evaluation_question(question)
        pre = self.bundle.get("precomputed", {}).get(q["id"]) if q else None
        if pre:
            return {
                "question": question,
                "config": pre["config"],
                "sources": pre["sources"],
                "answers": pre["answers"],
                "quota": self.gate.status(ip),
                "demo_mode": False,
                "message": "Evaluation question: answers from the evaluation run, nothing generated now.",
            }

        hits = self.retrievers[LIVE_STRATEGY].search(LIVE_METHOD, question, budget_chars=self.budget_chars)
        sources = [_source(i, h) for i, h in enumerate(hits, 1)]
        response: dict[str, Any] = {
            "question": question,
            "config": f"{LIVE_STRATEGY}_{LIVE_METHOD}",
            "sources": sources,
            "answers": [],
            "demo_mode": True,
            "message": None,
        }
        reason = "Live answers are off on this deployment." if self.llm is None else self.gate.acquire(ip)
        if reason is None and self.llm is not None:
            reason = self._live_answer(question, hits, response)
        if reason:
            response["message"] = f"{reason} {DEMO_HINT}"
        response["quota"] = self.gate.status(ip)
        return response

    def _live_answer(self, question: str, hits: Sequence[Hit], response: dict[str, Any]) -> str | None:
        from ragk8s.groq_client import DailyQuotaReached, GroqError

        assert self.llm is not None
        try:
            gen = self.llm(question, hits)
        except DailyQuotaReached:
            self.gate.pause.set_until_midnight("The model's daily limit is reached.")
            return "The model's daily limit is reached."
        except Busy:
            return "The model is busy. Try again in a minute."
        except GroqError:
            return "The model didn't answer."
        check = check_citations(gen.answer, len(hits))
        response["answers"] = [
            {
                "model": self.llm_model or gen.model or "",
                "text": gen.answer,
                "is_refusal": is_refusal(gen.answer),
                "cited": list(check.cited),
                "origin": "live",
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        ]
        response["demo_mode"] = False
        return None

    # --- /api/search -----------------------------------------------------------------

    def search(self, query: str, strategies: Sequence[str], methods: Sequence[str]) -> dict[str, Any]:
        q = self.evaluation_question(query)
        doc_id = q.get("source_doc_id") if q else None
        cells = []
        for strategy in dict.fromkeys(strategies):
            for method in dict.fromkeys(methods):
                cells.append(self._cell(strategy, method, query, q, doc_id))
        return {
            "query": query.strip(),
            "budget_chars": self.budget_chars,
            "evaluation_question": q is not None,
            "source_url": q.get("source_url") if q else None,
            "cells": cells,
        }

    def _cell(
        self, strategy: str, method: str, query: str, q: Mapping[str, Any] | None, doc_id: str | None
    ) -> dict[str, Any]:
        cell: dict[str, Any] = {"strategy": strategy, "method": method}
        if method == "hybrid_rerank":
            sources = self.bundle.get("rerank", {}).get(strategy, {}).get(q["id"]) if q else None
            cell.update(
                available=sources is not None,
                origin="precomputed",
                sources=sources or [],
                context_chars=sum(len(s["text"]) for s in sources or []),
                milliseconds=None,
                source_rank=_source_rank(sources or [], doc_id),
            )
            return cell
        started = self._clock()
        hits = self.retrievers[strategy].search(method, query, budget_chars=self.budget_chars)
        sources = [_source(i, h) for i, h in enumerate(hits, 1)]
        cell.update(
            available=True,
            origin="live",
            sources=sources,
            context_chars=sum(len(s["text"]) for s in sources),
            milliseconds=round(1000 * (self._clock() - started), 1),
            source_rank=_source_rank(sources, doc_id),
        )
        return cell
