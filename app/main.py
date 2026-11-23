"""FastAPI app: the JSON API under /api and, when built, the React front-end.

uvicorn app.main:app --port 8080
"""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from ragk8s import __version__, config

from . import schemas
from .engine import Engine, NoReranker, RetrieverLike, groq_llm
from .quota import GcsCounterStore, MemoryCounterStore, QuotaGate, VisitorLimiter
from .settings import Settings


def client_ip(request: Request) -> str:
    """Visitor address as seen by Cloud Run's front end.

    Google's front end appends the address it received the request from to
    X-Forwarded-For, so the last entry is the one a visitor can't forge; earlier
    entries are whatever the client sent.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    entries = [e.strip() for e in forwarded.split(",") if e.strip()]
    if entries:
        return entries[-1]
    return request.client.host if request.client else "unknown"


def build_engine(settings: Settings) -> Engine:
    """Production wiring: indexes from data/index, ONNX fp32 embedder, Groq if a key is set."""
    from ragk8s.onnx_models import OnnxEmbedder
    from ragk8s.retrieval import Retriever

    bundle = json.loads((settings.data_dir / "bundle.json").read_text(encoding="utf-8"))
    embedder = OnnxEmbedder(variant="fp32", threads=settings.threads)
    retrievers: dict[str, RetrieverLike] = {
        s: Retriever(s, embed_model=cast(Any, embedder), cross_encoder=cast(Any, NoReranker()))
        for s in config.STRATEGIES
    }
    store = GcsCounterStore(settings.quota_bucket) if settings.quota_bucket else MemoryCounterStore()
    gate = QuotaGate(store, settings.daily_limit, VisitorLimiter(settings.visitor_hourly_limit), settings.live_llm)
    llm, model = groq_llm(settings.groq_api_key) if settings.live_llm else (None, "")
    return Engine(bundle, retrievers, gate, llm=llm, llm_model=model)


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or Settings()
    state: dict[str, Engine] = {}
    lock = threading.Lock()

    def get_engine() -> Engine:
        if "engine" not in state:
            with lock:  # first requests may arrive together: load the indexes once
                if "engine" not in state:
                    state["engine"] = engine or build_engine(settings)
        return state["engine"]

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        get_engine()  # load before serving: Cloud Run sends traffic only once startup is done
        yield

    app = FastAPI(
        lifespan=lifespan,
        title="DocTruth API",
        version=__version__,
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url=None,
    )

    @app.get("/api/health", response_model=schemas.Health)
    def health() -> dict[str, Any]:
        e = get_engine()
        return {
            "status": "ok",
            "version": __version__,
            "strategies": list(e.retrievers),
            "live_llm": e.llm is not None,
            "bundle_created_at": e.bundle.get("created_at"),
        }

    @app.get("/api/quota", response_model=schemas.Quota)
    def quota(request: Request) -> dict[str, Any]:
        return get_engine().gate.status(client_ip(request))

    @app.get("/api/examples", response_model=list[schemas.ExampleQuestion])
    def examples() -> list[dict[str, Any]]:
        return get_engine().examples()

    @app.post("/api/ask", response_model=schemas.AskResponse)
    def ask(body: schemas.AskRequest, request: Request) -> dict[str, Any]:
        return get_engine().ask(body.question, client_ip(request))

    @app.post("/api/search", response_model=schemas.SearchResponse)
    def search(body: schemas.SearchRequest, request: Request) -> dict[str, Any]:
        e = get_engine()
        if not e.search_limiter.try_use(client_ip(request)):
            raise HTTPException(status_code=429, detail="Too many searches this hour. Try again later.")
        return e.search(body.query, body.strategies, body.methods)

    @app.get("/api/scores", response_model=schemas.ScoresResponse)
    def scores() -> dict[str, Any]:
        b = get_engine().bundle
        s = b.get("scores", {})
        return {
            "live_config": b.get("live_config", "fixed_hybrid"),
            "best_config": b.get("best_config", "fixed_hybrid_rerank"),
            "retrieval": s.get("retrieval", []),
            "generation": s.get("generation"),
            "onnx_parity": s.get("onnx_parity"),
            "measured_on": b.get("created_at"),
        }

    static = settings.static_dir
    if (static / "index.html").exists():

        @app.get("/{path:path}", include_in_schema=False)
        def front(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(status_code=404)
            target = (static / path).resolve()
            if path and target.is_file() and static.resolve() in target.parents:
                return FileResponse(target)
            return FileResponse(static / "index.html")  # client-side routes

    return app


app = create_app()
