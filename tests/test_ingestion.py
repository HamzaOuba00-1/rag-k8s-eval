from __future__ import annotations

from pathlib import Path

import pytest

from ragk8s import config, ingestion


@pytest.mark.parametrize(
    ("path", "url"),
    [
        ("content/en/docs/concepts/_index.md", "https://kubernetes.io/docs/concepts/"),
        ("content/en/docs/concepts/workloads/pods/_index.md", "https://kubernetes.io/docs/concepts/workloads/pods/"),
        (
            "content/en/docs/tasks/run-application/run-stateless-application-deployment.md",
            "https://kubernetes.io/docs/tasks/run-application/run-stateless-application-deployment/",
        ),
        ("content/en/docs/tutorials/md-in-name.md", "https://kubernetes.io/docs/tutorials/md-in-name/"),
    ],
)
def test_build_doc_url(path: str, url: str) -> None:
    assert ingestion.build_doc_url(path) == url


def test_build_doc_url_rejects_non_doc_paths() -> None:
    with pytest.raises(ValueError):
        ingestion.build_doc_url("content/en/blog/post.md")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_walk_order_lists_files_per_directory_breadth_of_children(tmp_path: Path) -> None:
    root = tmp_path / "concepts"
    for rel in ["b.md", "_index.md", "zeta/x.md", "Alpha/y.md", "Alpha/deep/z.md", "zeta/notes.txt"]:
        _write(root / rel, "x")
    got = [p.relative_to(root).as_posix() for p in ingestion.walk_markdown(root)]
    # root files, then Alpha, zeta (children of root), then Alpha/deep.
    assert got == ["b.md", "_index.md", "Alpha/y.md", "zeta/x.md", "Alpha/deep/z.md"]


def test_parse_export_verify_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clone = tmp_path / "clone"
    body = "Real content. " * 20
    _write(clone / "content/en/docs/concepts/a.md", f"---\ntitle: Page A\n---\n{body}")
    _write(clone / "content/en/docs/concepts/empty.md", "---\ntitle: Empty\n---\nshort")
    _write(clone / "content/en/docs/tasks/b.md", body)
    (clone / "content/en/docs/tutorials").mkdir(parents=True)

    docs = ingestion.parse_corpus(clone)
    # The skipped file still consumes an id, like the original snapshot.
    assert [(d.doc_id, d.title) for d in docs] == [("doc_0000", "Page A"), ("doc_0002", "b")]

    raw = tmp_path / "raw"
    ingestion.export(docs, raw_dir=raw, commit="abc")
    assert ingestion.verify(raw) == []
    assert ingestion.compare_with_snapshot(docs, raw_dir=raw) == []

    (raw / "doc_0002.md").write_text("tampered", encoding="utf-8")
    assert ingestion.verify(raw) == ["doc_0002: content does not match manifest"]


def test_pinned_commit_is_a_full_sha() -> None:
    assert len(config.K8S_WEBSITE_COMMIT) == 40
