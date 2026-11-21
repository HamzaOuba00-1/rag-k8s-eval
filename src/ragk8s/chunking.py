"""Chunking: three strategies compared in the evaluation.

1. fixed     - fixed-size character windows with overlap (naive baseline).
2. structure - split on Markdown headings (## and ###); long sections are
               re-split with a fixed window, each piece keeping its heading.
3. semantic  - consecutive sentences are merged while the cosine similarity
               of adjacent sentence embeddings stays above a threshold.

Output: data/processed/chunks_<strategy>.jsonl, one chunk per line.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from . import config
from .config import Strategy
from .utils import write_jsonl

EmbedFn = Callable[[list[str]], npt.NDArray[np.float32]]
"""Maps sentences to L2-normalised embeddings, one row per sentence."""

_HEADING_RE = re.compile(r"(?m)^(#{2,3}\s+.+)$")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_fixed(text: str, chunk_size: int = config.FIXED_CHUNK_SIZE, overlap: int = config.FIXED_OVERLAP) -> list[str]:
    """Fixed-size character windows; consecutive windows share ``overlap`` chars."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("need chunk_size > 0 and 0 <= overlap < chunk_size")
    step = chunk_size - overlap
    return [text[start : start + chunk_size] for start in range(0, len(text), step)]


def chunk_structure(
    text: str,
    max_chunk_size: int = config.STRUCTURE_MAX_CHUNK_SIZE,
    overlap: int = config.STRUCTURE_OVERLAP,
) -> list[str]:
    """Split on ``##``/``###`` headings, never mixing two sections in one chunk.

    Sections longer than ``max_chunk_size`` are re-split with a fixed window;
    every piece is prefixed with its heading and stays within the limit.
    """
    chunks: list[str] = []
    header = ""
    for part in _HEADING_RE.split(text):
        if not part.strip():
            continue
        if _HEADING_RE.fullmatch(part):
            header = part.strip()
            continue
        body = part.strip()
        section = f"{header}\n{body}" if header else body
        if len(section) <= max_chunk_size:
            chunks.append(section)
            continue
        prefix = f"{header}\n" if header else ""
        window = max(max_chunk_size - len(prefix), overlap + 1)
        chunks.extend(prefix + piece for piece in chunk_fixed(body, chunk_size=window, overlap=overlap))
    return chunks or [text]


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter (newlines are flattened first)."""
    flat = text.replace("\n", " ")
    return [s.strip() for s in _SENTENCE_END_RE.split(flat) if s.strip()]


def chunk_semantic(
    text: str,
    embed: EmbedFn,
    threshold: float = config.SEMANTIC_THRESHOLD,
    max_chunk_size: int = config.SEMANTIC_MAX_CHUNK_SIZE,
    overlap: int = config.STRUCTURE_OVERLAP,
) -> list[str]:
    """Merge adjacent sentences while their cosine similarity is >= ``threshold``.

    A "sentence" longer than ``max_chunk_size`` (typically a code block with no
    punctuation) is split with a fixed window instead.
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text]
    embeddings = embed(sentences)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for i, sentence in enumerate(sentences):
        if len(sentence) > max_chunk_size:
            if current:
                chunks.append(" ".join(current))
            current, current_len = [], 0
            chunks.extend(chunk_fixed(sentence, chunk_size=max_chunk_size, overlap=overlap))
            continue
        if not current:
            current, current_len = [sentence], len(sentence)
            continue
        similarity = float(np.dot(embeddings[i], embeddings[i - 1]))
        if similarity >= threshold and current_len + len(sentence) <= max_chunk_size:
            current.append(sentence)
            current_len += len(sentence)
        else:
            chunks.append(" ".join(current))
            current, current_len = [sentence], len(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(
    strategy: Strategy, docs: Sequence[dict[str, Any]], chunk_fn: Callable[[str], list[str]]
) -> list[dict[str, Any]]:
    """Apply ``chunk_fn`` to every document and attach metadata."""
    records: list[dict[str, Any]] = []
    for doc in docs:
        for piece in chunk_fn(doc["text"]):
            piece = piece.strip()
            if len(piece) < config.MIN_CHUNK_CHARS:
                continue
            records.append(
                {
                    "chunk_id": f"{strategy}_{len(records):05d}",
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "source_url": doc["source_url"],
                    "strategy": strategy,
                    "text": piece,
                }
            )
    return records


def sentence_embedder(model_name: str = config.EMBEDDING_MODEL) -> EmbedFn:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(sentences: list[str]) -> npt.NDArray[np.float32]:
        return np.asarray(model.encode(sentences, show_progress_bar=False, normalize_embeddings=True), dtype=np.float32)

    return embed


def chunk_function(strategy: Strategy, embed: EmbedFn | None = None) -> Callable[[str], list[str]]:
    if strategy == "fixed":
        return chunk_fixed
    if strategy == "structure":
        return chunk_structure
    if embed is None:
        raise ValueError("semantic chunking needs an embedding function")
    return lambda text: chunk_semantic(text, embed)


def chunks_path(strategy: Strategy, processed_dir: Path = config.PROCESSED_DIR) -> Path:
    return processed_dir / f"chunks_{strategy}.jsonl"


def run(
    docs: Sequence[dict[str, Any]], strategies: Sequence[Strategy] = config.STRATEGIES
) -> dict[str, dict[str, float]]:
    """Chunk the corpus with each strategy and write the JSONL files. Returns stats."""
    embed = sentence_embedder() if "semantic" in strategies else None
    stats: dict[str, dict[str, float]] = {}
    for strategy in strategies:
        records = build_chunks(strategy, docs, chunk_function(strategy, embed))
        write_jsonl(chunks_path(strategy), records)
        lengths = np.array([len(r["text"]) for r in records])
        stats[strategy] = {
            "n_chunks": len(records),
            "mean_chars": float(lengths.mean()),
            "median_chars": float(np.median(lengths)),
            "max_chars": int(lengths.max()),
        }
        print(f"{strategy:<10} {len(records):>6} chunks, mean {lengths.mean():.0f} chars -> {chunks_path(strategy)}")
    return stats
