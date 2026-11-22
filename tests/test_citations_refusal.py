from __future__ import annotations

import pytest

from ragk8s.citations import check_citations, cited_numbers, sentence_coverage
from ragk8s.refusal import REFUSAL_SENTENCE, is_refusal, refusal_stats


def test_cited_numbers_formats() -> None:
    assert cited_numbers("A [1]. B [2][3]. C [1, 4]; D [5;6].") == [1, 2, 3, 1, 4, 5, 6]


def test_brackets_that_are_not_citations_are_ignored() -> None:
    assert cited_numbers("Use `kubectl get pods [-o wide]` or a list [a, b].") == []


def test_check_citations_valid_and_invalid() -> None:
    check = check_citations("Pods are units [1][2]. Nodes run them [7].", n_sources=5)
    assert check.cited == (1, 2)
    assert check.invalid == (7,)
    assert check.n_markers == 3
    assert check.has_citation and not check.all_valid
    assert check.coverage == 1.0


def test_coverage_ignores_code_blocks_and_counts_uncited_sentences() -> None:
    answer = (
        "Run this command to scale [1].\n```\nkubectl scale deployment/web --replicas=3\n```\n"
        "It changes replicas quickly."
    )
    assert sentence_coverage(answer) == pytest.approx(0.5)


def test_no_citation() -> None:
    check = check_citations("No sources here at all.", n_sources=5)
    assert not check.has_citation and check.coverage == 0.0


@pytest.mark.parametrize(
    "answer",
    [
        REFUSAL_SENTENCE,
        '"I cannot find this information in the provided documentation."',
        "I can't find this information in the provided documentation.",
        "The provided excerpts do not contain this information.",
    ],
)
def test_refusals_detected(answer: str) -> None:
    assert is_refusal(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "A Deployment manages ReplicaSets [1].",
        "Je ne trouve pas cette information dans la documentation fournie.",
        "I cannot find a reason to use hostNetwork here; use a Service instead [2].",
    ],
)
def test_non_refusals(answer: str) -> None:
    assert not is_refusal(answer)


def test_refusal_stats() -> None:
    records = [
        {"category": "impossible", "answer": REFUSAL_SENTENCE},
        {"category": "impossible", "answer": "Docker Swarm is enabled with ... [1]"},
        {"category": "factual", "answer": "Pods are the smallest unit [1]."},
        {"category": "factual", "answer": REFUSAL_SENTENCE},
        {"category": "factual", "answer": "Services expose Pods [2]."},
        {"category": "factual", "answer": "Nodes run Pods [3]."},
    ]
    stats = refusal_stats(records)
    assert stats.refusal_rate == 0.5
    assert stats.false_refusal_rate == 0.25


def test_refusal_stats_empty_and_unknown_category() -> None:
    assert refusal_stats([]).refusal_rate is None
    with pytest.raises(ValueError):
        refusal_stats([{"category": "other", "answer": ""}])
