from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ragk8s import indexing, retrieval
from ragk8s.retrieval import Hit, Retriever, fill_budget, pool_by_chars, reciprocal_rank_fusion, top_indices


def test_rrf_scores_and_order() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)
    assert [item for item, _ in fused] == ["b", "a", "d", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert dict(fused)["c"] == pytest.approx(1 / 63)


def test_rrf_ties_are_broken_by_first_appearance() -> None:
    fused = reciprocal_rank_fusion([["x", "y"], ["y", "x"]])
    assert [item for item, _ in fused] == ["x", "y"]


def test_rrf_empty() -> None:
    assert reciprocal_rank_fusion([]) == []


def test_top_indices_is_stable() -> None:
    assert top_indices(np.array([0.5, 2.0, 0.5, 1.0]), 3) == [1, 3, 0]


def test_tokenize() -> None:
    assert indexing.tokenize("kube-proxy runs on Node_1!") == ["kube", "proxy", "runs", "on", "node", "1"]


class FakeEmbedder:
    """Bag-of-words embedding over a tiny vocabulary."""

    vocab = ("pod", "service", "volume", "node")

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        rows = np.array([[t.lower().count(w) + 1e-3 for w in self.vocab] for t in texts], dtype=np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float("volume" in text) for _q, text in pairs]


@pytest.fixture()
def tiny_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import faiss

    chunks = [
        {"chunk_id": f"fixed_{i:05d}", "doc_id": f"doc_{i:04d}", "title": f"T{i}", "source_url": "u", "text": t}
        for i, t in enumerate(
            ["A pod is the smallest unit.", "A service exposes pods.", "A volume stores data.", "A node runs pods."]
        )
    ]
    directory = tmp_path / "fixed"
    directory.mkdir()
    (directory / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8")
    (directory / "bm25_tokens.json").write_text(json.dumps([indexing.tokenize(c["text"]) for c in chunks]))
    vectors = FakeEmbedder().encode([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(directory / "dense.faiss"))
    monkeypatch.setattr(retrieval, "index_dir", lambda strategy: tmp_path / strategy)
    return directory


def test_retriever_methods_on_tiny_index(tiny_index: Path) -> None:
    r = Retriever("fixed", embed_model=FakeEmbedder(), cross_encoder=FakeCrossEncoder())  # type: ignore[arg-type]
    assert r.bm25("service", top_k=1)[0].chunk["doc_id"] == "doc_0001"
    assert r.dense("volume", top_k=1)[0].chunk["doc_id"] == "doc_0002"
    hybrid = r.hybrid("node pods", top_k=4, pool=4)
    assert [h.rank for h in hybrid] == [1, 2, 3, 4]
    assert len({h.chunk_id for h in hybrid}) == 4
    reranked = r.hybrid_rerank("pods", top_k=2, pool_chars=10_000, min_candidates=4, max_candidates=4)
    assert reranked[0].chunk["doc_id"] == "doc_0002"
    assert [h.rank for h in reranked] == [1, 2]


def test_dense_ignores_missing_ids_when_top_k_exceeds_corpus(tiny_index: Path) -> None:
    r = Retriever("fixed", embed_model=FakeEmbedder())  # type: ignore[arg-type]
    assert len(r.dense("pod", top_k=10)) == 4


def test_unknown_method(tiny_index: Path) -> None:
    r = Retriever("fixed", embed_model=FakeEmbedder())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        r.search("magic", "q")  # type: ignore[arg-type]


def _hit(text: str, rank: int) -> Hit:
    return Hit({"chunk_id": f"c{rank}", "text": text}, 1.0 / rank, rank)


def test_fill_budget_skips_chunks_that_do_not_fit_and_renumbers() -> None:
    hits = [_hit("a" * 40, 1), _hit("b" * 70, 2), _hit("c" * 50, 3), _hit("d" * 10, 4)]
    selected = fill_budget(hits, budget_chars=100)
    assert [h.chunk["chunk_id"] for h in selected] == ["c1", "c3", "c4"]
    assert [h.rank for h in selected] == [1, 2, 3]
    assert sum(len(h.chunk["text"]) for h in selected) <= 100
    assert [h.rank for h in hits] == [1, 2, 3, 4]  # input untouched


def test_fill_budget_respects_max_chunks() -> None:
    hits = [_hit("x", i) for i in range(1, 30)]
    assert len(fill_budget(hits, budget_chars=1000, max_chunks=20)) == 20


def test_pool_by_chars() -> None:
    hits = [_hit("x" * 100, i) for i in range(1, 50)]
    assert len(pool_by_chars(hits, pool_chars=1000, min_hits=5, max_hits=40)) == 10
    assert len(pool_by_chars(hits, pool_chars=100, min_hits=5, max_hits=40)) == 5
    assert len(pool_by_chars(hits, pool_chars=10**6, min_hits=5, max_hits=40)) == 40


def test_search_applies_budget(tiny_index: Path) -> None:
    r = Retriever("fixed", embed_model=FakeEmbedder())  # type: ignore[arg-type]
    context = r.search("bm25", "pod", budget_chars=50)
    assert sum(len(h.chunk["text"]) for h in context) <= 50
    assert len(context) == 2
