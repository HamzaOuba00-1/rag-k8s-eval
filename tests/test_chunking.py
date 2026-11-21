from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from ragk8s.chunking import build_chunks, chunk_fixed, chunk_semantic, chunk_structure, split_sentences


def test_fixed_windows_cover_text_with_overlap() -> None:
    text = "abcdefghij" * 25  # 250 chars
    chunks = chunk_fixed(text, chunk_size=100, overlap=20)
    assert [len(c) for c in chunks] == [100, 100, 90, 10]
    assert chunks[0][-20:] == chunks[1][:20]
    # Dropping the overlap of every window but the last rebuilds the text.
    assert "".join(c[:80] for c in chunks[:-1]) + chunks[-1] == text


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_fixed_rejects_invalid_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_fixed("text", chunk_size=size, overlap=overlap)


def test_structure_splits_on_level_2_and_3_headings_only() -> None:
    text = "Intro paragraph.\n## Install\nRun the installer.\n### Linux\nUse apt.\n#### Deep\nStill Linux.\n# Title"
    chunks = chunk_structure(text, max_chunk_size=1000)
    assert chunks == [
        "Intro paragraph.",
        "## Install\nRun the installer.",
        "### Linux\nUse apt.\n#### Deep\nStill Linux.\n# Title",
    ]


def test_structure_long_section_pieces_keep_heading_and_respect_limit() -> None:
    body = "x" * 1000
    chunks = chunk_structure(f"## Heading\n{body}", max_chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(c.startswith("## Heading\n") for c in chunks)
    assert all(len(c) <= 300 for c in chunks)


def test_structure_without_headings_returns_whole_text() -> None:
    assert chunk_structure("plain text") == ["plain text"]


def test_split_sentences() -> None:
    assert split_sentences("One. Two!\nThree? Four") == ["One.", "Two!", "Three?", "Four"]


def _fake_embed(vectors: dict[str, list[float]]):  # type: ignore[no-untyped-def]
    def embed(sentences: list[str]) -> npt.NDArray[np.float32]:
        rows = np.array([vectors[s] for s in sentences], dtype=np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    return embed


def test_semantic_groups_similar_adjacent_sentences() -> None:
    embed = _fake_embed({"Pods run.": [1, 0], "Pods restart.": [1, 0.1], "Cats purr.": [0, 1]})
    assert chunk_semantic("Pods run. Pods restart. Cats purr.", embed, threshold=0.5) == [
        "Pods run. Pods restart.",
        "Cats purr.",
    ]


def test_semantic_splits_oversized_sentence_with_fixed_windows() -> None:
    long = "y" * 50
    embed = _fake_embed({"Short.": [1, 0], long: [1, 0]})
    chunks = chunk_semantic(f"Short. {long}", embed, threshold=0.5, max_chunk_size=20, overlap=5)
    assert chunks[0] == "Short."
    assert all(len(c) <= 20 for c in chunks[1:])


def test_build_chunks_drops_tiny_pieces_and_numbers_sequentially() -> None:
    docs = [{"doc_id": "doc_0001", "title": "T", "source_url": "u", "text": "ignored"}]
    records = build_chunks("fixed", docs, lambda _t: ["tiny", "a" * 60, "b" * 60])
    assert [r["chunk_id"] for r in records] == ["fixed_00000", "fixed_00001"]
    assert records[0]["doc_id"] == "doc_0001"
