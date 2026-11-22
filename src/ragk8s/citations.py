"""Parsing and checking of [n] citations in generated answers."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.S)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_SENTENCE_CHARS = 15


@dataclass(frozen=True)
class CitationCheck:
    cited: tuple[int, ...]
    """Distinct valid source numbers, sorted."""
    invalid: tuple[int, ...]
    """Distinct cited numbers that do not match any provided source."""
    n_markers: int
    """Number of individual citations (``[1][2]`` and ``[1, 2]`` both count 2)."""
    coverage: float
    """Share of answer sentences that carry at least one citation (0 if no sentence)."""

    @property
    def has_citation(self) -> bool:
        return self.n_markers > 0

    @property
    def all_valid(self) -> bool:
        return not self.invalid


def cited_numbers(answer: str) -> list[int]:
    """All citation numbers in order of appearance, with repetitions."""
    return [int(n) for group in _CITATION_RE.findall(answer) for n in re.split(r"\s*[,;]\s*", group)]


def sentence_coverage(answer: str) -> float:
    prose = _CODE_BLOCK_RE.sub(" ", answer)
    sentences = [s.strip() for s in _SENTENCE_RE.split(prose) if len(s.strip()) >= _MIN_SENTENCE_CHARS]
    if not sentences:
        return 0.0
    return sum(1 for s in sentences if _CITATION_RE.search(s)) / len(sentences)


def check_citations(answer: str, n_sources: int) -> CitationCheck:
    numbers = cited_numbers(answer)
    distinct = sorted(set(numbers))
    return CitationCheck(
        cited=tuple(n for n in distinct if 1 <= n <= n_sources),
        invalid=tuple(n for n in distinct if not 1 <= n <= n_sources),
        n_markers=len(numbers),
        coverage=sentence_coverage(answer),
    )
