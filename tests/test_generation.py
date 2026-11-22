from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ragk8s import config
from ragk8s.generation import GROQ_PARAMS, build_context, groq_backend
from ragk8s.groq_client import ChatResult
from ragk8s.retrieval import Hit


class FakeGroq:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def chat(self, model: str, messages: Sequence[Mapping[str, str]], max_tokens: int, **params: Any) -> ChatResult:
        self.kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens, **params}
        return ChatResult("<think>hidden</think> Pods are units [1].", "qwen/qwen3.8-27b", 1500, 20, 1520, 0.8)


def _hits() -> list[Hit]:
    return [Hit({"chunk_id": "c", "title": "Pods", "text": "A Pod is the smallest unit.", "source_url": "u"}, 1.0, 1)]


def test_groq_backend_disables_reasoning_and_strips_think_tags() -> None:
    fake = FakeGroq()
    backend = groq_backend("qwen/qwen3.8-27b", client=fake)
    gen = backend.generate("What is a Pod?", _hits())
    assert gen.answer == "Pods are units [1]."
    assert (gen.prompt_tokens, gen.completion_tokens) == (1500, 20)
    assert fake.kwargs["reasoning_effort"] == "none" and fake.kwargs["temperature"] == 0.0
    assert fake.kwargs["max_tokens"] == config.GENERATION_MAX_TOKENS
    assert fake.kwargs["messages"][0]["role"] == "system"
    assert backend.params == GROQ_PARAMS and backend.slug == "qwen3.8-27b"


def test_context_numbering_matches_citations() -> None:
    hits = _hits() * 3
    context = build_context(hits)
    assert context.startswith("[1] Pods\n") and "\n\n[3] Pods\n" in context
