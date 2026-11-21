from __future__ import annotations

import pytest

from ragk8s.cli import build_parser


def test_parser_defaults() -> None:
    args = build_parser().parse_args(["run"])
    assert (args.strategy, args.method, args.top_k, args.limit) == ("structure", "hybrid_rerank", 5, None)


def test_parser_rejects_unknown_strategy() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["index", "--strategies", "random"])
