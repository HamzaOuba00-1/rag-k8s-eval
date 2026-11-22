from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from ragk8s import kaggle, pipeline
from ragk8s.pipeline import RunConfig


def test_export_contains_contexts_questions_and_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "RETRIEVAL_DIR", tmp_path / "retrieval")
    questions = tmp_path / "questions.jsonl"
    questions.write_text('{"question": "Q?"}\n', encoding="utf-8")
    monkeypatch.setattr(kaggle, "QUESTIONS_PATH", questions)
    cfg = RunConfig("fixed", "hybrid_rerank")
    path = pipeline.retrieval_path(cfg)
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    path.with_suffix(".meta.json").write_text("{}", encoding="utf-8")

    out = kaggle.export("contexts", [cfg], commit="abc123", out_dir=tmp_path / "export")
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        manifest = json.loads(z.read("manifest.json"))
    assert {"results/retrieval/fixed_hybrid_rerank.jsonl", "eval/questions.jsonl", "manifest.json"} <= names
    assert manifest["commit"] == "abc123" and manifest["configs"] == ["fixed_hybrid_rerank"]


def _kaggle_output(root: Path, answer: str) -> None:
    run = root / "results" / "runs" / "qwen2.5-7b-instruct" / "run_fixed_hybrid_rerank.jsonl"
    run.parent.mkdir(parents=True)
    run.write_text('{"id": "q"}\n', encoding="utf-8")
    cache = root / ".cache" / "generations" / "abc.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"answer": answer}), encoding="utf-8")


def test_import_copies_runs_and_cache_and_never_overwrites_a_cached_answer(tmp_path: Path) -> None:
    runs, cache = tmp_path / "runs", tmp_path / "cache"
    first = tmp_path / "out1"
    _kaggle_output(first, "A")
    stats = kaggle.import_outputs(first, runs_dir=runs, cache_dir=cache)
    assert stats["copied"] == 2 and not stats["conflicts"]
    assert (runs / "qwen2.5-7b-instruct" / "run_fixed_hybrid_rerank.jsonl").exists()

    again = kaggle.import_outputs(first, runs_dir=runs, cache_dir=cache)
    assert again["identical"] == 2 and again["copied"] == 0

    other = tmp_path / "out2"
    _kaggle_output(other, "B")
    zipped = tmp_path / "output.zip"
    with zipfile.ZipFile(zipped, "w") as z:
        for p in other.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(other))
    clash = kaggle.import_outputs(zipped, runs_dir=runs, cache_dir=cache)
    assert len(clash["conflicts"]) == 1
    assert json.loads((cache / "abc.json").read_text(encoding="utf-8"))["answer"] == "A"


def test_import_rejects_unrelated_folder(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        kaggle.import_outputs(tmp_path, runs_dir=tmp_path / "r", cache_dir=tmp_path / "c")
