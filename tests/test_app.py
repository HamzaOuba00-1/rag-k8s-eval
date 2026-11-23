from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.engine import Busy, Engine, normalise
from app.main import create_app
from app.quota import MemoryCounterStore, QuotaGate, VisitorLimiter, next_midnight
from app.settings import Settings
from ragk8s.generation import Generation
from ragk8s.groq_client import DailyQuotaReached
from ragk8s.retrieval import Hit

NOW = datetime(2026, 11, 23, 10, 30, tzinfo=UTC)


def _chunk(i: int, doc: str) -> dict[str, Any]:
    return {
        "chunk_id": f"fixed_{i:05d}",
        "doc_id": doc,
        "title": f"Page {doc}",
        "source_url": f"https://kubernetes.io/docs/{doc}/",
        "text": f"text {i}",
    }


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def search(self, method: Any, query: str, budget_chars: int = 5000) -> list[Hit]:
        self.calls.append((method, query))
        if method == "hybrid_rerank":
            raise AssertionError("never reranked live")
        return [Hit(_chunk(1, "doc_b"), 0.5, 1), Hit(_chunk(2, "doc_a"), 0.4, 2)]


BUNDLE = {
    "created_at": "2026-11-25T10:00:00+00:00",
    "live_config": "fixed_hybrid",
    "best_config": "fixed_hybrid_rerank",
    "budget_chars": 5000,
    "questions": [
        {
            "id": "q1",
            "question": "How do I drain a node?",
            "category": "factual",
            "source_doc_id": "doc_a",
            "source_url": "https://kubernetes.io/docs/doc_a/",
        },
    ],
    "precomputed": {
        "q1": {
            "config": "fixed_hybrid_rerank",
            "sources": [
                {
                    "n": 1,
                    "title": "A",
                    "url": "u",
                    "doc_id": "doc_a",
                    "chunk_id": "fixed_00002",
                    "text": "t",
                    "score": 6.1,
                }
            ],
            "answers": [
                {
                    "model": "qwen3.8-27b",
                    "text": "Use kubectl drain [1].",
                    "is_refusal": False,
                    "cited": [1],
                    "origin": "precomputed",
                    "generated_at": "2026-11-24T09:00:00+00:00",
                }
            ],
        }
    },
    "rerank": {
        "fixed": {
            "q1": [{"n": 1, "title": "A", "url": "u", "doc_id": "doc_a", "chunk_id": "c", "text": "abc", "score": 6.1}]
        }
    },
    "scores": {"retrieval": [], "generation": None, "onnx_parity": None},
}


def _llm(answer: str = "Drain it with kubectl drain [1][7].") -> Any:
    calls: list[str] = []

    def run(question: str, hits: Sequence[Hit]) -> Generation:
        calls.append(question)
        return Generation(
            answer=answer, model="qwen/qwen3.8-27b", prompt_tokens=1, completion_tokens=1, total_seconds=0.1
        )

    run.calls = calls  # type: ignore[attr-defined]
    return run


def _engine(llm: Any = None, daily: int = 2, hourly: int = 5, live: bool = True) -> Engine:
    gate = QuotaGate(MemoryCounterStore(), daily, VisitorLimiter(hourly), live_llm=live, now=lambda: NOW)
    retrievers = {s: FakeRetriever() for s in ("fixed", "structure", "semantic")}
    return Engine(BUNDLE, retrievers, gate, llm=llm, llm_model="qwen3.8-27b")


def _client(engine: Engine) -> TestClient:
    return TestClient(create_app(Settings(), engine=engine))


def test_evaluation_question_is_served_precomputed_without_quota_or_llm() -> None:
    llm = _llm()
    engine = _engine(llm)
    body = _client(engine).post("/api/ask", json={"question": "  how do I drain a NODE "}).json()
    assert body["config"] == "fixed_hybrid_rerank"
    assert body["answers"][0]["origin"] == "precomputed"
    assert llm.calls == []
    assert body["quota"]["remaining_today"] == 2


def test_free_question_gets_a_live_answer_with_valid_citations_only() -> None:
    engine = _engine(_llm())
    body = _client(engine).post("/api/ask", json={"question": "What is a Pod?"}).json()
    assert body["config"] == "fixed_hybrid"
    assert [s["n"] for s in body["sources"]] == [1, 2]
    answer = body["answers"][0]
    assert answer["origin"] == "live" and answer["cited"] == [1]  # [7] is not a source
    assert body["demo_mode"] is False
    assert body["quota"]["remaining_today"] == 1


def test_daily_limit_switches_to_demo_mode_but_keeps_live_sources() -> None:
    client = _client(_engine(_llm(), daily=1))
    client.post("/api/ask", json={"question": "What is a Pod?"})
    body = client.post("/api/ask", json={"question": "What is a Service?"}).json()
    assert body["answers"] == [] and body["demo_mode"] is True
    assert body["sources"] and "used up" in body["message"]
    assert body["quota"]["live_enabled"] is False


def test_visitor_hourly_limit_is_per_visitor() -> None:
    client = _client(_engine(_llm(), daily=10, hourly=1))
    first = {"x-forwarded-for": "203.0.113.5"}
    client.post("/api/ask", json={"question": "What is a Pod?"}, headers=first)
    blocked = client.post("/api/ask", json={"question": "What is a Job?"}, headers=first).json()
    other = client.post("/api/ask", json={"question": "What is a Job?"}, headers={"x-forwarded-for": "198.51.100.7"})
    assert "this hour" in blocked["message"]
    assert other.json()["answers"]


def test_groq_daily_limit_pauses_live_answers_until_midnight() -> None:
    def exhausted(question: str, hits: Sequence[Hit]) -> Generation:
        raise DailyQuotaReached("tpd")

    engine = _engine(exhausted, daily=10)
    client = _client(engine)
    body = client.post("/api/ask", json={"question": "What is a Pod?"}).json()
    assert body["answers"] == [] and "daily limit" in body["message"]
    quota = client.get("/api/quota").json()
    assert quota["live_enabled"] is False and engine.gate.pause.until == next_midnight(NOW)


def test_busy_model_is_reported_without_pausing() -> None:
    def busy(question: str, hits: Sequence[Hit]) -> Generation:
        raise Busy("wait 30 s")

    engine = _engine(busy, daily=10)
    body = _client(engine).post("/api/ask", json={"question": "What is a Pod?"}).json()
    assert "busy" in body["message"] and not engine.gate.pause.active


def test_no_key_means_demo_mode() -> None:
    body = _client(_engine(None, live=False)).post("/api/ask", json={"question": "What is a Pod?"}).json()
    assert body["demo_mode"] is True and "off" in body["message"]


def test_search_runs_cheap_methods_live_and_serves_rerank_precomputed() -> None:
    engine = _engine()
    body = (
        _client(engine)
        .post(
            "/api/search",
            json={
                "query": "How do I drain a node?",
                "strategies": ["fixed", "semantic"],
                "methods": ["hybrid", "hybrid_rerank"],
            },
        )
        .json()
    )
    cells = {(c["strategy"], c["method"]): c for c in body["cells"]}
    assert body["evaluation_question"] is True
    assert cells[("fixed", "hybrid")]["origin"] == "live" and cells[("fixed", "hybrid")]["source_rank"] == 2
    assert cells[("fixed", "hybrid_rerank")]["origin"] == "precomputed"
    assert cells[("fixed", "hybrid_rerank")]["source_rank"] == 1
    assert cells[("semantic", "hybrid_rerank")]["available"] is False


def test_search_rerank_is_unavailable_for_a_free_query() -> None:
    body = _client(_engine()).post("/api/search", json={"query": "anything else", "methods": ["hybrid_rerank"]}).json()
    assert all(not c["available"] and c["sources"] == [] for c in body["cells"])


def test_request_validation_and_openapi() -> None:
    client = _client(_engine())
    assert client.post("/api/ask", json={"question": "a"}).status_code == 422
    paths = client.get("/api/openapi.json").json()["paths"]
    assert {"/api/ask", "/api/search", "/api/scores", "/api/quota", "/api/examples", "/api/health"} <= set(paths)


def test_scores_never_invent_missing_results() -> None:
    body = _client(_engine()).get("/api/scores").json()
    assert body["generation"] is None and body["onnx_parity"] is None
    assert body["live_config"] == "fixed_hybrid"


def test_visitor_ids_are_keyed_hashes_not_ips() -> None:
    limiter = VisitorLimiter(1, key=b"k")
    vid = limiter.visitor_id("203.0.113.5")
    assert "203" not in vid and len(vid) == 16
    assert VisitorLimiter(1, key=b"other").visitor_id("203.0.113.5") != vid


def test_normalise_ignores_case_spaces_and_question_mark() -> None:
    assert normalise(" How  do I drain a NODE? ") == normalise("how do i drain a node")
