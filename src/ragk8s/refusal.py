"""Detection of explicit refusals and refusal metrics.

The generation prompt asks for one canonical sentence when the excerpts do not
contain the answer, which makes detection a string match rather than a guess.
A few close paraphrases are also accepted, because a 7B model does not always
copy the sentence verbatim; everything else counts as an answer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

REFUSAL_SENTENCE = "I cannot find this information in the provided documentation."

_ACCEPTED_PATTERNS = (
    "i cannot find this information in the provided documentation",
    "i cannot find this information in the provided excerpts",
    "i could not find this information in the provided documentation",
    "the provided documentation does not contain this information",
    "the provided excerpts do not contain this information",
)


def _normalise(text: str) -> str:
    text = text.lower().replace("\u2019", "'").replace("can't", "cannot").replace("can not", "cannot")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", text)).strip()


def is_refusal(answer: str) -> bool:
    normalised = _normalise(answer)
    return any(pattern in normalised for pattern in _ACCEPTED_PATTERNS)


@dataclass(frozen=True)
class RefusalStats:
    n_impossible: int
    n_refused_impossible: int
    n_factual: int
    n_refused_factual: int

    @property
    def refusal_rate(self) -> float | None:
        """Share of out-of-corpus questions correctly refused (higher is better)."""
        return self.n_refused_impossible / self.n_impossible if self.n_impossible else None

    @property
    def false_refusal_rate(self) -> float | None:
        """Share of answerable questions wrongly refused (lower is better)."""
        return self.n_refused_factual / self.n_factual if self.n_factual else None


def refusal_stats(records: Iterable[Mapping[str, Any]]) -> RefusalStats:
    """Count refusals per category. Each record needs ``category`` and ``answer``."""
    counts = {"impossible": [0, 0], "factual": [0, 0]}
    for record in records:
        category = record["category"]
        if category not in counts:
            raise ValueError(f"unknown category: {category}")
        counts[category][0] += 1
        counts[category][1] += is_refusal(record["answer"])
    return RefusalStats(*counts["impossible"], *counts["factual"])
