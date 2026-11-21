"""Retrieval over one chunking strategy: BM25, dense, hybrid (RRF) and hybrid + rerank."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from . import config
from .config import Method, Strategy
from .indexing import index_dir, tokenize
from .utils import read_jsonl

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer


@dataclass
class Hit:
    chunk: dict[str, Any]
    score: float
    rank: int
    rerank_score: float | None = field(default=None)

    @property
    def chunk_id(self) -> str:
        return str(self.chunk["chunk_id"])


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = config.RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(d) = sum over lists of 1 / (k + rank(d)), rank from 1.

    Ties are broken by first appearance (earlier lists first), so the output
    is deterministic.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    order = {item: i for i, item in enumerate(scores)}
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


def top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    """Indices of the ``top_k`` highest scores; ties keep index order."""
    return [int(i) for i in np.argsort(-scores, kind="stable")[:top_k]]


class Retriever:
    """Loads the indexes of one strategy; models are loaded on first use."""

    def __init__(
        self,
        strategy: Strategy,
        embed_model: SentenceTransformer | None = None,
        cross_encoder: CrossEncoder | None = None,
    ) -> None:
        import faiss
        from rank_bm25 import BM25Okapi

        directory = index_dir(strategy)
        self.strategy = strategy
        self.chunks = read_jsonl(directory / "chunks.jsonl")
        tokens = json.loads((directory / "bm25_tokens.json").read_text(encoding="utf-8"))
        self._bm25 = BM25Okapi(tokens)
        self._faiss = faiss.read_index(str(directory / "dense.faiss"))
        if self._faiss.ntotal != len(self.chunks):
            raise RuntimeError(f"{directory}: index and chunk metadata are out of sync, rebuild the index")
        self._embed_model = embed_model
        self._cross_encoder = cross_encoder

    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
        return self._embed_model

    @property
    def cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(config.RERANKER_MODEL)
        return self._cross_encoder

    def bm25(self, query: str, top_k: int = config.TOP_K) -> list[Hit]:
        scores = np.asarray(self._bm25.get_scores(tokenize(query)))
        return [Hit(self.chunks[i], float(scores[i]), r) for r, i in enumerate(top_indices(scores, top_k), start=1)]

    def dense(self, query: str, top_k: int = config.TOP_K) -> list[Hit]:
        vec = np.asarray(self.embed_model.encode([query], normalize_embeddings=True), dtype=np.float32)
        scores, ids = self._faiss.search(vec, top_k)
        pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]
        return [Hit(self.chunks[i], s, r) for r, (i, s) in enumerate(pairs, start=1)]

    def hybrid(self, query: str, top_k: int = config.TOP_K, pool: int = config.HYBRID_CANDIDATE_POOL) -> list[Hit]:
        bm25_hits = self.bm25(query, pool)
        dense_hits = self.dense(query, pool)
        lookup = {h.chunk_id: h.chunk for h in (*bm25_hits, *dense_hits)}
        fused = reciprocal_rank_fusion([[h.chunk_id for h in bm25_hits], [h.chunk_id for h in dense_hits]])
        return [Hit(lookup[cid], score, r) for r, (cid, score) in enumerate(fused[:top_k], start=1)]

    def hybrid_rerank(
        self, query: str, top_k: int = config.TOP_K, candidates: int = config.RERANK_CANDIDATES
    ) -> list[Hit]:
        hits = self.hybrid(query, top_k=candidates)
        scores = self.cross_encoder.predict([(query, h.chunk["text"]) for h in hits])
        for hit, s in zip(hits, scores, strict=True):
            hit.rerank_score = float(s)
        # Stable sort: equal rerank scores keep their fused order.
        ranked = sorted(hits, key=lambda h: -(h.rerank_score or 0.0))[:top_k]
        for r, hit in enumerate(ranked, start=1):
            hit.rank = r
        return ranked

    def search(self, method: Method, query: str, top_k: int = config.TOP_K) -> list[Hit]:
        if method == "bm25":
            return self.bm25(query, top_k)
        if method == "dense":
            return self.dense(query, top_k)
        if method == "hybrid":
            return self.hybrid(query, top_k)
        if method == "hybrid_rerank":
            return self.hybrid_rerank(query, top_k)
        raise ValueError(f"unknown retrieval method: {method}")
