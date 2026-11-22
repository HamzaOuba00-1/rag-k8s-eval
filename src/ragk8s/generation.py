"""Answer generation with mandatory [n] citations (Ollama).

The retrieved chunks are numbered [1]..[k] in the prompt. The model may only
cite those numbers; the number -> (title, URL) mapping is attached afterwards,
so a cited source cannot be invented. When the excerpts do not contain the
answer, the model must reply with one canonical sentence (refusal.py).

Decoding is deterministic (temperature 0, fixed seed, explicit context size),
so that a run can be repeated and compared.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from . import config
from .refusal import REFUSAL_SENTENCE
from .retrieval import Hit

PROMPT_VERSION = "v2-en"

SYSTEM_PROMPT = f"""You are a technical assistant that answers questions about Kubernetes \
using ONLY the numbered documentation excerpts provided by the user.

Rules:
1. Use only information stated in the excerpts. Never add facts from general knowledge.
2. After every factual statement, cite the excerpt(s) that support it with their number \
in square brackets, for example [1] or [2][3].
3. If the excerpts do not contain the answer, reply with exactly this sentence and nothing else:
"{REFUSAL_SENTENCE}"
4. Answer in English, concisely."""

USER_TEMPLATE = """Documentation excerpts:

{context}

Question: {question}"""

OLLAMA_OPTIONS: dict[str, Any] = {
    "temperature": 0.0,
    "seed": config.SEED,
    "num_ctx": config.GENERATION_NUM_CTX,
    "num_predict": config.GENERATION_MAX_TOKENS,
}


@dataclass(frozen=True)
class Generation:
    answer: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_seconds: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_context(hits: Sequence[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.chunk['title']}\n{h.chunk['text']}" for i, h in enumerate(hits, start=1))


def build_messages(question: str, hits: Sequence[Hit]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=build_context(hits), question=question)},
    ]


def sources(hits: Sequence[Hit]) -> list[dict[str, Any]]:
    return [
        {
            "n": i,
            "chunk_id": h.chunk["chunk_id"],
            "doc_id": h.chunk["doc_id"],
            "title": h.chunk["title"],
            "url": h.chunk["source_url"],
        }
        for i, h in enumerate(hits, start=1)
    ]


def _field(obj: Any, name: str) -> Any:
    """Read a field from an ollama response object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def generate(question: str, hits: Sequence[Hit], model: str = config.OLLAMA_MODEL) -> Generation:
    import ollama

    response = ollama.chat(model=model, messages=build_messages(question, hits), options=OLLAMA_OPTIONS)
    total_ns = _field(response, "total_duration")
    return Generation(
        answer=str(_field(_field(response, "message"), "content")).strip(),
        model=model,
        prompt_tokens=_field(response, "prompt_eval_count"),
        completion_tokens=_field(response, "eval_count"),
        total_seconds=total_ns / 1e9 if total_ns else None,
    )


def model_digest(model: str = config.OLLAMA_MODEL) -> str | None:
    """Digest of the local Ollama model, recorded with each run (None if unavailable)."""
    import ollama

    try:
        listing = ollama.list()
    except Exception:  # Ollama not running: the run will fail later with a clear error.
        return None
    for entry in _field(listing, "models") or []:
        if _field(entry, "model") == model or _field(entry, "name") == model:
            digest = _field(entry, "digest")
            return str(digest) if digest else None
    return None
