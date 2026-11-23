"""API contract of the DocTruth web app.

These Pydantic models are the single source of truth for the HTTP API. FastAPI
publishes them as OpenAPI (``/api/openapi.json``) and the React front-end
generates its TypeScript types from that schema, so both sides cannot drift apart.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ragk8s.config import METHODS, STRATEGIES, Method, Strategy

# --- shared -------------------------------------------------------------------------


class Source(BaseModel):
    """An excerpt given to the generator; ``n`` is the number used in citations."""

    n: int = Field(description="Citation number, [n] in the answer")
    title: str
    url: str = Field(description="Public page on kubernetes.io")
    doc_id: str
    chunk_id: str
    text: str
    score: float = Field(description="Retriever score (rerank score for hybrid_rerank)")


class Quota(BaseModel):
    live_enabled: bool = Field(description="False in demo mode: no live LLM answer until the reset")
    remaining_today: int = Field(description="Live answers still available today (app-wide)")
    daily_limit: int
    visitor_remaining_this_hour: int
    resets_at: str = Field(description="ISO 8601 time (UTC) when the daily counter resets")
    reason: str | None = Field(default=None, description="Why live answers are paused, if they are")


# --- /api/ask -------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Answer(BaseModel):
    model: str
    text: str = Field(description="Answer with [n] citations, or the refusal sentence")
    is_refusal: bool
    cited: list[int] = Field(description="Valid citation numbers used in the answer")
    origin: Literal["live", "precomputed"]
    generated_at: str | None = None


class AskResponse(BaseModel):
    question: str
    config: str = Field(description="Chunking and retrieval used, e.g. fixed_hybrid")
    sources: list[Source]
    answers: list[Answer] = Field(description="Live answer, or precomputed answers of the evaluated models")
    quota: Quota
    demo_mode: bool
    message: str | None = Field(default=None, description="Shown to the user, e.g. quota reached")


class ExampleQuestion(BaseModel):
    id: str
    question: str
    category: Literal["factual", "impossible"]
    source_url: str | None = None


# --- /api/search (Compare / Playground, no LLM) -----------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    strategies: list[Strategy] = Field(default_factory=lambda: list(STRATEGIES))
    methods: list[Method] = Field(default_factory=lambda: list(METHODS))


class SearchCell(BaseModel):
    strategy: Strategy
    method: Method
    available: bool = Field(description="False when this cell can't be computed live for this query")
    origin: Literal["live", "precomputed"]
    sources: list[Source]
    context_chars: int
    milliseconds: float | None = Field(description="Measured server time; null for precomputed cells")
    source_rank: int | None = Field(
        default=None, description="Rank of the question's source page, for evaluation questions only"
    )


class SearchResponse(BaseModel):
    query: str
    budget_chars: int
    evaluation_question: bool = Field(description="The query is one of the evaluation questions")
    source_url: str | None = Field(default=None, description="Page the evaluation question was written from")
    cells: list[SearchCell]


# --- /api/scores ------------------------------------------------------------------------


class Estimate(BaseModel):
    mean: float
    low: float
    high: float
    n: int


class RetrievalRow(BaseModel):
    config: str
    chunking: Strategy
    retrieval: Method
    n: int
    mrr: Estimate
    hit_at_1: Estimate
    hit_at_5: Estimate
    context_chars: float
    mrr_diff_vs_best: Estimate | None = Field(description="Paired bootstrap difference with the best config")


class ScoresResponse(BaseModel):
    """Measured results only; a part not measured yet is null."""

    live_config: str = Field(description="Configuration used for live answers")
    best_config: str = Field(description="Best configuration of the evaluation")
    retrieval: list[RetrievalRow]
    generation: dict[str, object] | None = Field(description="results/summary.json, once the judge run is complete")
    onnx_parity: dict[str, object] | None
    measured_on: str | None = Field(description="Date the bundle was built from the result files (UTC)")


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    strategies: list[Strategy]
    live_llm: bool = Field(description="A Groq key is configured")
    bundle_created_at: str | None
