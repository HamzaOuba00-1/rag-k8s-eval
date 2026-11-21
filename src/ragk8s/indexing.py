"""Indexing: one BM25 index and one dense (FAISS) index per chunking strategy.

Layout of data/index/<strategy>/:
  chunks.jsonl      chunk metadata, in index order (row i = vector i = BM25 doc i)
  bm25_tokens.json  tokenised corpus; BM25 is rebuilt from it at load time
                    (a few seconds, and no pickle to trust or keep compatible)
  dense.faiss       inner-product index over L2-normalised embeddings (= cosine)
  index_meta.json   embedding model, sizes and the SHA-256 of the source chunks

The hybrid retriever has no index of its own: it fuses BM25 and dense rankings
at query time (see retrieval.py).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

from . import config
from .chunking import chunks_path
from .config import Strategy
from .utils import read_jsonl, sha256_file, write_json

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens (shared by indexing and querying)."""
    return _TOKEN_RE.findall(text.lower())


def index_dir(strategy: Strategy, root: Path = config.INDEX_DIR) -> Path:
    return root / strategy


def build(strategy: Strategy, embed_model_name: str = config.EMBEDDING_MODEL, batch_size: int = 64) -> Path:
    import faiss
    from sentence_transformers import SentenceTransformer

    src = chunks_path(strategy)
    chunks = read_jsonl(src)
    out = index_dir(strategy)
    out.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(src, out / "chunks.jsonl")
    tokens = [tokenize(c["text"]) for c in chunks]
    (out / "bm25_tokens.json").write_text(json.dumps(tokens), encoding="utf-8")

    model = SentenceTransformer(embed_model_name)
    embeddings = model.encode(
        [c["text"] for c in chunks],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(out / "dense.faiss"))

    write_json(
        out / "index_meta.json",
        {
            "strategy": strategy,
            "n_chunks": len(chunks),
            "embedding_model": embed_model_name,
            "embedding_dim": int(vectors.shape[1]),
            "chunks_sha256": sha256_file(src),
        },
    )
    print(f"{strategy:<10} {len(chunks):>6} chunks indexed -> {out}")
    return out
