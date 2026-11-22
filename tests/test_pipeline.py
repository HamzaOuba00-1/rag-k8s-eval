from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from ragk8s import pipeline
from ragk8s.generation import Generation
from ragk8s.pipeline import RunConfig, generate_run, generation_key, question_id
from ragk8s.refusal import REFUSAL_SENTENCE
from ragk8s.retrieval import Hit


def test_run_config_parse() -> None:
    assert RunConfig.parse("semantic:dense") == RunConfig("semantic", "dense")
    assert RunConfig.parse("semantic:dense").name == "semantic_dense"
    with pytest.raises(ValueError):
        RunConfig.parse("semantic")
    assert len(pipeline.ALL_CONFIGS) == 12


def test_question_id_is_stable_and_whitespace_insensitive() -> None:
    assert question_id("What is a Pod?") == question_id("  What is a Pod? ")
    assert question_id("What is a Pod?") != question_id("What is a Node?")


def test_generation_key_depends_on_contexts_and_model() -> None:
    base = generation_key("q", ["a", "b"], "m", "d1")
    assert base == generation_key("q", ["a", "b"], "m", "d1")
    assert base != generation_key("q", ["b", "a"], "m", "d1")
    assert base != generation_key("q", ["a", "b"], "m", "d2")


def _write_retrieval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunConfig:
    monkeypatch.setattr(pipeline, "RETRIEVAL_DIR", tmp_path / "retrieval")
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path / "runs")
    cfg = RunConfig("fixed", "bm25")
    hit = {
        "rank": 1,
        "score": 1.0,
        "rerank_score": None,
        "chunk_id": "fixed_00000",
        "doc_id": "doc_0001",
        "title": "T",
        "source_url": "u",
        "text": "Pods are the smallest unit.",
    }
    items = [
        {
            "id": "q_1",
            "question": "What is a Pod?",
            "category": "factual",
            "ground_truth": "g",
            "source_doc_id": "doc_0001",
            "hits": [hit],
        },
        {
            "id": "q_2",
            "question": "Docker Swarm?",
            "category": "impossible",
            "ground_truth": "n/a",
            "source_doc_id": None,
            "hits": [hit],
        },
    ]
    path = pipeline.retrieval_path(cfg)
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    path.with_suffix(".meta.json").write_text("{}", encoding="utf-8")
    return cfg


def test_generate_run_caches_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_retrieval(tmp_path, monkeypatch)
    calls: list[str] = []

    def fake(question: str, hits: Sequence[Hit]) -> Generation:
        calls.append(question)
        answer = REFUSAL_SENTENCE if "Swarm" in question else "A Pod is the smallest unit [1]."
        return Generation(answer, "m", 10, 5, 1.5)

    cache = tmp_path / "cache"
    out = generate_run(cfg, model="m", generate_fn=fake, digest="d", cache_dir=cache)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["is_refusal"] for r in records] == [False, True]
    assert records[0]["cited"] == [1] and records[0]["citation_coverage"] == 1.0
    assert len(calls) == 2

    # Second invocation: everything comes from the cache.
    generate_run(cfg, model="m", generate_fn=fake, digest="d", cache_dir=cache)
    assert len(calls) == 2
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["n_questions"] == 2 and meta["generation_seconds_total"] == 3.0


def test_generate_run_interrupted_then_resumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_retrieval(tmp_path, monkeypatch)
    monkeypatch.setattr("ragk8s.pipeline.time.sleep", lambda _s: None)
    calls: list[str] = []

    def flaky(question: str, hits: Sequence[Hit]) -> Generation:
        calls.append(question)
        if "Swarm" in question and len(calls) < 10:
            raise ConnectionError("Ollama down")
        return Generation("ok [1]", "m", 1, 1, 1.0)

    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="cached"):
        generate_run(cfg, model="m", generate_fn=flaky, digest="d", cache_dir=cache)
    assert len(list(cache.glob("*.json"))) == 1  # the first answer survived the failure

    calls.clear()
    calls.extend(["x"] * 10)  # Ollama is back
    generate_run(cfg, model="m", generate_fn=flaky, digest="d", cache_dir=cache)
    assert calls[10:] == ["Docker Swarm?"]  # only the missing answer was generated
