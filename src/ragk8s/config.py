"""Central configuration: paths, model names, seeds and pipeline parameters.

Every value that influences a result lives here, so that a run can be fully
described by this module plus the package version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

# --- Paths (relative to the repository root, where commands are run) -------

DATA_DIR: Final = Path("data")
RAW_DIR: Final = DATA_DIR / "raw"
PROCESSED_DIR: Final = DATA_DIR / "processed"
INDEX_DIR: Final = DATA_DIR / "index"
CLONE_DIR: Final = DATA_DIR / "_k8s_website_clone"
EVAL_DIR: Final = Path("eval")
RESULTS_DIR: Final = Path("results")
CACHE_DIR: Final = Path(".cache")

# --- Corpus ------------------------------------------------------------------

K8S_WEBSITE_REPO: Final = "https://github.com/kubernetes/website.git"
# Upstream commit of the frozen snapshot in data/raw/ (2026-08-19, merge of
# PR #57060). Re-ingesting at this commit reproduces data/raw/ byte for byte.
K8S_WEBSITE_COMMIT: Final = "5184b9b24b288709ed36b4003f53dc1530eda2f3"
DOC_SECTIONS: Final = (
    "content/en/docs/concepts",
    "content/en/docs/tasks",
    "content/en/docs/tutorials",
)
MIN_DOC_CHARS: Final = 200  # shorter files are redirects or empty index pages

# --- Chunking ------------------------------------------------------------------

Strategy = Literal["fixed", "structure", "semantic"]
STRATEGIES: Final[tuple[Strategy, ...]] = ("fixed", "structure", "semantic")

FIXED_CHUNK_SIZE: Final = 1000
FIXED_OVERLAP: Final = 150
STRUCTURE_MAX_CHUNK_SIZE: Final = 1500
STRUCTURE_OVERLAP: Final = 100
SEMANTIC_THRESHOLD: Final = 0.55
SEMANTIC_MAX_CHUNK_SIZE: Final = 1500
MIN_CHUNK_CHARS: Final = 50
CLEAN_MARKUP: Final = True  # render Hugo shortcodes to text before chunking (textclean.py)

# --- Retrieval -------------------------------------------------------------------

Method = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
METHODS: Final[tuple[Method, ...]] = ("bm25", "dense", "hybrid", "hybrid_rerank")

EMBEDDING_MODEL: Final = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL: Final = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K: Final = 60
# Every strategy gets the same context budget in characters (not in chunks):
# chunks are taken in rank order while they fit, so semantic chunking (~290
# chars per chunk) is not starved compared with fixed/structure (~850-950).
CONTEXT_BUDGET_CHARS: Final = 5000
MAX_CONTEXT_CHUNKS: Final = 20  # hard cap on numbered excerpts in the prompt
CANDIDATE_DEPTH: Final = 60  # ranked candidates per retriever before budgeting / fusion
# The cross-encoder rescoring pool is also sized in characters (4x the budget),
# within [RERANK_MIN_CANDIDATES, RERANK_MAX_CANDIDATES] chunks.
RERANK_POOL_CHARS: Final = 4 * CONTEXT_BUDGET_CHARS
RERANK_MIN_CANDIDATES: Final = 20
RERANK_MAX_CANDIDATES: Final = 80

# --- Generation ------------------------------------------------------------------

OLLAMA_MODEL: Final = "qwen2.5:7b-instruct"
GENERATION_NUM_CTX: Final = 4096  # tokens; a 5,000-char context (~1,300 tokens) fits with margin
GENERATION_MAX_TOKENS: Final = 512
SEED: Final = 42
