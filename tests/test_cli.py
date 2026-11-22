from __future__ import annotations

import pytest

from ragk8s.cli import build_parser


def test_parser_generate_configs() -> None:
    args = build_parser().parse_args(["generate", "--configs", "structure:hybrid_rerank", "fixed:bm25", "--limit", "3"])
    assert args.configs == ["structure:hybrid_rerank", "fixed:bm25"]
    assert args.limit == 3 and not args.all


def test_parser_configs_and_all_are_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["retrieve", "--all", "--configs", "fixed:bm25"])


def test_parser_rejects_unknown_strategy() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["index", "--strategies", "random"])
