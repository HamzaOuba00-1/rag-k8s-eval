"""Corpus ingestion: Kubernetes documentation -> data/raw/.

The corpus is fetched at a pinned upstream commit (config.K8S_WEBSITE_COMMIT)
with a shallow, sparse, blob-filtered checkout, so only the English
concepts/tasks/tutorials Markdown files are downloaded.

For each Markdown file, the YAML front matter is removed and the body is
written to data/raw/doc_XXXX.md. data/raw/manifest.json keeps the original
path, the public URL (used for citations) and a SHA-256 of the content, and
data/raw/corpus.json records the upstream commit.

Document ids are positional: doc_XXXX is the index of the file in the walk
order below, *including* skipped (near-empty) files. The walk order mirrors
how the first snapshot was produced (Python 3.12 ``Path.rglob`` on NTFS), so
that re-ingesting reproduces the frozen snapshot and the doc ids referenced
by the gold questions in eval/.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import frontmatter

from . import config
from .utils import read_json_list, sha256_text, write_json

BASE_URL = "https://kubernetes.io"


@dataclass(frozen=True)
class RawDoc:
    doc_id: str
    title: str
    source_path: str
    source_url: str
    text: str

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "n_chars": len(self.text),
            "sha256": sha256_text(self.text),
        }


# --- Fetching ----------------------------------------------------------------


def _git(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def fetch_corpus(clone_dir: Path = config.CLONE_DIR, commit: str = config.K8S_WEBSITE_COMMIT) -> Path:
    """Shallow sparse checkout of kubernetes/website at ``commit``."""
    if (clone_dir / ".git").exists():
        print(f"{clone_dir} already exists, skipping download.")
        return clone_dir
    clone_dir.mkdir(parents=True, exist_ok=True)
    _git(["init", "--quiet"], cwd=clone_dir)
    _git(["remote", "add", "origin", config.K8S_WEBSITE_REPO], cwd=clone_dir)
    _git(["sparse-checkout", "set", "--no-cone", *[f"/{s}/" for s in config.DOC_SECTIONS]], cwd=clone_dir)
    _git(["fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", commit], cwd=clone_dir)
    _git(["-c", "advice.detachedHead=false", "checkout", "--quiet", "FETCH_HEAD"], cwd=clone_dir)
    return clone_dir


def _on_rm_error(func: Callable[..., Any], path: str, _exc: object) -> None:
    # Git marks some object files read-only on Windows.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_clone(clone_dir: Path = config.CLONE_DIR) -> None:
    if clone_dir.exists():
        if sys.version_info >= (3, 12):
            shutil.rmtree(clone_dir, onexc=_on_rm_error)
        else:  # pragma: no cover
            shutil.rmtree(clone_dir, onerror=_on_rm_error)


# --- Parsing -------------------------------------------------------------------


def walk_markdown(root: Path) -> list[Path]:
    """List ``*.md`` files under ``root`` in a deterministic, OS-independent order.

    Directories are enumerated like Python 3.12 ``Path.rglob``: ``root``
    first, then the children of every directory visited by a top-down walk;
    files are listed per directory. Names are compared case-insensitively,
    like NTFS directory listings.
    """

    def entries(d: Path) -> list[Path]:
        return sorted(d.iterdir(), key=lambda p: p.name.upper())

    dirs = [root]

    def visit(d: Path) -> None:
        subdirs = [p for p in entries(d) if p.is_dir()]
        dirs.extend(subdirs)
        for sub in subdirs:
            visit(sub)

    visit(root)
    return [p for d in dirs for p in entries(d) if p.is_file() and p.suffix == ".md"]


def build_doc_url(source_path: str) -> str:
    """``content/en/docs/concepts/foo/_index.md`` -> ``https://kubernetes.io/docs/concepts/foo/``."""
    parts = PurePosixPath(source_path).parts
    if "docs" not in parts or not source_path.endswith(".md"):
        raise ValueError(f"Not a documentation page: {source_path}")
    rel = list(parts[parts.index("docs") :])
    stem = PurePosixPath(rel[-1]).stem
    rel = rel[:-1] if stem == "_index" else [*rel[:-1], stem]
    return f"{BASE_URL}/{'/'.join(rel)}/"


def parse_corpus(clone_dir: Path) -> list[RawDoc]:
    files: list[Path] = []
    for section in config.DOC_SECTIONS:
        files.extend(walk_markdown(clone_dir / section))

    docs = []
    for i, path in enumerate(files):
        post = frontmatter.load(str(path))
        text = post.content.strip()
        if len(text) < config.MIN_DOC_CHARS:
            continue
        source_path = path.relative_to(clone_dir).as_posix()
        docs.append(
            RawDoc(
                doc_id=f"doc_{i:04d}",
                title=str(post.get("title", path.stem)),
                source_path=source_path,
                source_url=build_doc_url(source_path),
                text=text,
            )
        )
    return docs


# --- Export / verification -------------------------------------------------------


def export(docs: list[RawDoc], raw_dir: Path = config.RAW_DIR, commit: str = config.K8S_WEBSITE_COMMIT) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        (raw_dir / f"{doc.doc_id}.md").write_text(doc.text, encoding="utf-8", newline="\n")
    write_json(raw_dir / "manifest.json", [d.manifest_entry() for d in docs])
    write_json(
        raw_dir / "corpus.json",
        {
            "source_repo": config.K8S_WEBSITE_REPO,
            "commit": commit,
            "sections": list(config.DOC_SECTIONS),
            "n_docs": len(docs),
            "n_chars": sum(len(d.text) for d in docs),
            "license": "CC BY 4.0 (see NOTICE)",
        },
    )


def load_raw_docs(raw_dir: Path = config.RAW_DIR) -> list[dict[str, Any]]:
    """Load the frozen corpus: manifest entries plus their ``text``."""
    docs = []
    for entry in read_json_list(raw_dir / "manifest.json"):
        text = (raw_dir / f"{entry['doc_id']}.md").read_text(encoding="utf-8")
        docs.append({**entry, "text": text.replace("\r\n", "\n")})
    return docs


def verify(raw_dir: Path = config.RAW_DIR) -> list[str]:
    """Check every document against the SHA-256 recorded in the manifest.

    Returns the list of problems (empty when the snapshot is intact).
    """
    problems = []
    for doc in load_raw_docs(raw_dir):
        expected = doc.get("sha256")
        if expected is None:
            problems.append(f"{doc['doc_id']}: no sha256 in manifest")
        elif sha256_text(doc["text"]) != expected:
            problems.append(f"{doc['doc_id']}: content does not match manifest")
    return problems


def compare_with_snapshot(docs: list[RawDoc], raw_dir: Path = config.RAW_DIR) -> list[str]:
    """Compare freshly parsed docs with the frozen snapshot, without writing anything."""
    snapshot = {d["doc_id"]: d for d in load_raw_docs(raw_dir)}
    fresh = {d.doc_id: d for d in docs}
    problems = [f"{i}: missing from fresh ingestion" for i in sorted(snapshot.keys() - fresh.keys())]
    problems += [f"{i}: not in snapshot" for i in sorted(fresh.keys() - snapshot.keys())]
    for doc_id in sorted(snapshot.keys() & fresh.keys()):
        old, new = snapshot[doc_id], fresh[doc_id]
        # Older manifests stored Windows-style paths.
        if old["text"] != new.text or old["source_path"].replace("\\", "/") != new.source_path:
            problems.append(f"{doc_id}: differs from snapshot")
    return problems
