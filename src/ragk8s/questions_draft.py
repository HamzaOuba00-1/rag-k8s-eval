"""Draft extra gold questions from whole pages (neutral with respect to chunking).

The first 55 factual questions were drafted from single *structure* chunks,
which can favour that chunking strategy. This module drafts additional
questions from whole pages that no existing question uses, so that results
can also be reported on a subset that no chunking strategy has seen.

- Pages are sampled with a fixed seed, proportionally to the three doc
  sections (concepts / tasks / tutorials), among pages with enough text.
- The drafting model is a different family from the generator under
  evaluation (default: openai/gpt-oss-120b on Groq), to limit self-preference.
- The model must return a verbatim evidence sentence; a draft whose evidence
  is not found in the page, or whose question refers to "this page", is
  rejected automatically. Accepted drafts are appended to eval/questions.jsonl
  with ``origin: page`` and ``reviewed: false``; the human review decides.
- Each page's draft is cached, so an interrupted run resumes.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import config
from .groq_client import DailyQuotaReached, GroqClient
from .ingestion import load_raw_docs
from .textclean import clean_markup
from .utils import read_jsonl, sha256_text, write_json, write_jsonl

DRAFT_MODEL = "openai/gpt-oss-120b"
DRAFT_CACHE = config.CACHE_DIR / "drafts"
QUESTIONS_PATH = config.EVAL_DIR / "questions.jsonl"
REJECTED_PATH = config.EVAL_DIR / "review" / "drafts_rejected.jsonl"
MIN_PAGE_CHARS = 1500
MAX_PAGE_CHARS = 6000
PROMPT_VERSION = "draft-v1"

SYSTEM_PROMPT = """You write evaluation questions for a question-answering system over the \
official Kubernetes documentation. Reply with a single JSON object and nothing else."""

USER_TEMPLATE = """Write ONE factual question about the documentation page below, with its answer.

Requirements:
- The answer is stated explicitly in the page and has exactly one correct form.
- The question is understandable on its own: never mention "this page", "the document", \
"the section", "the text" or "above".
- Prefer specific facts (a field, a default value, a command, a behaviour), not opinions or lists of many items.
- The answer is short (one or two sentences) and only says what the page says.
- "evidence" is ONE sentence copied verbatim from the page that contains the answer.

Return: {{"question": "...", "answer": "...", "evidence": "..."}}

Page title: {title}
Page URL: {url}

Page text:
\"\"\"
{text}
\"\"\""""

_SELF_REFERENCE_RE = re.compile(
    r"\b(this|the) (page|document|documentation page|section|text|excerpt|passage|article)\b|\babove\b", re.I
)


def section_of(source_path: str) -> str:
    parts = source_path.split("/")
    return parts[3] if len(parts) > 3 else "other"


def page_text(doc: dict[str, Any], max_chars: int = MAX_PAGE_CHARS) -> str:
    """Cleaned page text, cut at a paragraph boundary when longer than ``max_chars``."""
    text = clean_markup(doc["text"])
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n\n", 0, max_chars)
    return text[: cut if cut > max_chars // 2 else max_chars]


def sample_pages(
    docs: Sequence[dict[str, Any]], exclude: set[str], n: int, seed: int = config.SEED
) -> list[dict[str, Any]]:
    """Seeded sample of ``n`` unused pages, allocated to sections proportionally (largest remainder)."""
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        if doc["doc_id"] not in exclude and len(clean_markup(doc["text"])) >= MIN_PAGE_CHARS:
            pool[section_of(doc["source_path"])].append(doc)
    total = sum(len(v) for v in pool.values())
    if n > total:
        raise ValueError(f"only {total} eligible pages for {n} questions")
    sections = sorted(pool)
    quotas = {s: n * len(pool[s]) / total for s in sections}
    alloc = {s: int(quotas[s]) for s in sections}
    for s in sorted(sections, key=lambda s: -(quotas[s] - alloc[s]))[: n - sum(alloc.values())]:
        alloc[s] += 1
    rng = random.Random(seed)
    picked = [doc for s in sections for doc in rng.sample(sorted(pool[s], key=lambda d: d["doc_id"]), alloc[s])]
    return sorted(picked, key=lambda d: d["doc_id"])


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[`*_]", "", text)).strip().lower()


def validate(draft: dict[str, Any], page: str) -> str | None:
    """Return the reason a draft is rejected, or None when it passes the automatic checks."""
    for key in ("question", "answer", "evidence"):
        if not isinstance(draft.get(key), str) or not draft[key].strip():
            return f"missing {key}"
    question = draft["question"].strip()
    if not question.endswith("?") or not 15 <= len(question) <= 250:
        return "question format"
    if _SELF_REFERENCE_RE.search(question):
        return "question refers to the page itself"
    if len(draft["answer"]) > 400:
        return "answer too long"
    evidence = _normalise(draft["evidence"])
    if len(evidence) < 20 or evidence not in _normalise(page):
        return "evidence not found verbatim in the page"
    return None


def _parse(content: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def draft(
    n: int = 25,
    model: str = DRAFT_MODEL,
    seed: int = config.SEED,
    client: GroqClient | None = None,
    questions_path: Path = QUESTIONS_PATH,
    cache_dir: Path = DRAFT_CACHE,
    rejected_path: Path = REJECTED_PATH,
    replace: bool = False,
) -> dict[str, int]:
    # ``replace`` also discards *reviewed* page questions.
    """Draft questions for ``n`` sampled pages and append the accepted ones to the gold file."""
    questions = read_jsonl(questions_path)
    reviewed_page = [q for q in questions if q.get("origin") == "page" and q.get("reviewed")]
    if reviewed_page and not replace:
        raise FileExistsError(f"{len(reviewed_page)} reviewed page questions in {questions_path}; use --replace")
    # Unreviewed page drafts are rebuilt from the cache: rerunning resumes after a quota stop.
    questions = [q for q in questions if q.get("origin") != "page"]
    for q in questions:
        if q.get("category", "factual") == "factual":
            q.setdefault("origin", "structure_chunk")

    used = {q["source_doc_id"] for q in questions if q.get("source_doc_id")}
    pages = sample_pages(load_raw_docs(), used, n, seed)
    client = client or GroqClient()
    cache_dir.mkdir(parents=True, exist_ok=True)

    accepted, rejected = [], []
    tokens = 0
    try:
        for doc in pages:
            text = page_text(doc)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(title=doc["title"], url=doc["source_url"], text=text)},
            ]
            key = sha256_text(json.dumps({"m": model, "p": PROMPT_VERSION, "msg": messages}, sort_keys=True))[:16]
            cache_file = cache_dir / f"{doc['doc_id']}_{key}.json"
            if cache_file.exists():
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
            else:
                result = client.chat(
                    model,
                    messages,
                    max_tokens=1500,
                    temperature=0,
                    seed=seed,
                    reasoning_effort="low",
                    response_format={"type": "json_object"},
                )
                tokens += result.total_tokens
                cached = {"content": result.content, "total_tokens": result.total_tokens, "model": result.model}
                write_json(cache_file, cached)
            parsed = _parse(cached["content"])
            reason = validate(parsed, text)
            base = {"source_doc_id": doc["doc_id"], "source_title": doc["title"], "source_url": doc["source_url"]}
            if reason:
                rejected.append({**base, "reason": reason, "raw": cached["content"]})
                continue
            accepted.append(
                {
                    "question": parsed["question"].strip(),
                    "ground_truth": parsed["answer"].strip(),
                    **base,
                    "source_chunk_text": parsed["evidence"].strip(),
                    "category": "factual",
                    "origin": "page",
                    "drafted_by": cached.get("model", model),
                    "reviewed": False,
                }
            )
    except DailyQuotaReached as exc:
        print(f"Stopped: {exc}")
    factual = [q for q in questions if q.get("category", "factual") == "factual"]
    other = [q for q in questions if q.get("category", "factual") != "factual"]
    write_jsonl(questions_path, [*factual, *accepted, *other])
    if rejected:
        write_jsonl(rejected_path, rejected)
    return {"pages": len(pages), "accepted": len(accepted), "rejected": len(rejected), "new_tokens": tokens}
