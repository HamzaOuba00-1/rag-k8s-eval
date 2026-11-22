from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from ragk8s import questions_draft
from ragk8s.groq_client import ChatResult, DailyQuotaReached
from ragk8s.questions_draft import sample_pages, section_of, validate

PAGE = "Pods are the smallest deployable units. The default restartPolicy of a Pod is Always. " * 30


def _docs() -> list[dict[str, Any]]:
    docs = []
    for i in range(30):
        section = ["concepts", "concepts", "tasks"][i % 3] if i < 27 else "tutorials"
        docs.append(
            {
                "doc_id": f"doc_{i:04d}",
                "title": f"T{i}",
                "source_url": f"https://k/{i}/",
                "source_path": f"content/en/docs/{section}/p{i}.md",
                "text": PAGE,
            }
        )
    docs.append(
        {
            "doc_id": "doc_short",
            "title": "S",
            "source_url": "u",
            "source_path": "content/en/docs/tasks/s.md",
            "text": "too short",
        }
    )
    return docs


def test_section_of() -> None:
    assert section_of("content/en/docs/tasks/a/b.md") == "tasks"


def test_sample_pages_is_seeded_proportional_and_excludes() -> None:
    docs = _docs()
    a = sample_pages(docs, exclude={"doc_0000"}, n=10, seed=1)
    assert a == sample_pages(docs, exclude={"doc_0000"}, n=10, seed=1)
    ids = {d["doc_id"] for d in a}
    assert "doc_0000" not in ids and "doc_short" not in ids and len(ids) == 10
    sections = [section_of(d["source_path"]) for d in a]
    assert sections.count("concepts") == 6 and sections.count("tasks") == 3 and sections.count("tutorials") == 1
    with pytest.raises(ValueError):
        sample_pages(docs, exclude=set(), n=100)


GOOD = {
    "question": "What is the default restartPolicy of a Pod?",
    "answer": "Always.",
    "evidence": "The default restartPolicy of a Pod is Always.",
}


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({}, None),
        ({"question": "What does this page say about Pods?"}, "question refers to the page itself"),
        ({"question": "Default policy"}, "question format"),
        ({"evidence": "The default restartPolicy of a Pod is Never."}, "evidence not found verbatim in the page"),
        ({"answer": ""}, "missing answer"),
    ],
)
def test_validate(change: dict[str, str], reason: str | None) -> None:
    assert validate({**GOOD, **change}, PAGE) == reason


class FakeClient:
    def __init__(self, fail_after: int | None = None) -> None:
        self.calls = 0
        self.fail_after = fail_after

    def chat(self, model: str, messages: Sequence[Mapping[str, str]], max_tokens: int, **params: Any) -> ChatResult:
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise DailyQuotaReached("quota")
        self.calls += 1
        content = json.dumps(GOOD if self.calls % 2 else {**GOOD, "question": "What does this page say?"})
        return ChatResult(content, model, 900, 100, 1000, 0.5)


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(questions_draft, "load_raw_docs", _docs)
    qpath = tmp_path / "questions.jsonl"
    rows = [
        {"question": "Old?", "ground_truth": "g", "category": "factual", "source_doc_id": "doc_0001"},
        {"question": "Swarm?", "ground_truth": "n/a", "category": "impossible"},
    ]
    qpath.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return qpath


def test_draft_appends_accepted_and_resumes_from_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    qpath = _setup(tmp_path, monkeypatch)
    kwargs = {"questions_path": qpath, "cache_dir": tmp_path / "cache", "rejected_path": tmp_path / "rej.jsonl"}

    partial = questions_draft.draft(n=4, client=FakeClient(fail_after=2), **kwargs)  # type: ignore[arg-type]
    assert partial["accepted"] == 1 and partial["rejected"] == 1

    client = FakeClient()
    stats = questions_draft.draft(n=4, client=client, **kwargs)  # type: ignore[arg-type]
    assert client.calls == 2  # the two cached pages are not requested again
    assert stats["accepted"] + stats["rejected"] == 4

    rows = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["origin"] == "structure_chunk"
    assert rows[-1]["category"] == "impossible"
    page_rows = [r for r in rows if r.get("origin") == "page"]
    assert len(page_rows) == stats["accepted"] and all(not r["reviewed"] for r in page_rows)
    assert "doc_0001" not in {r["source_doc_id"] for r in page_rows}
