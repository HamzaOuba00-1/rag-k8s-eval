from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ragk8s.groq_client import (
    DailyQuotaReached,
    GroqClient,
    GroqError,
    Limits,
    UsageLedger,
    estimate_tokens,
    parse_duration,
)


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("7.66s", 7.66), ("2m59.56s", 179.56), ("1h2m3s", 3723.0), ("120ms", 0.12), ("30", 30.0), (None, 0.0), ("?", 0.0)],
)
def test_parse_duration(value: str | None, seconds: float) -> None:
    assert parse_duration(value) == pytest.approx(seconds)


def test_estimate_tokens() -> None:
    assert estimate_tokens([{"role": "user", "content": "x" * 350}], max_tokens=100) == 100 + 16 + 100


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _ok(total: int = 1000, headers: dict[str, str] | None = None) -> httpx.Response:
    body = {
        "model": "m",
        "choices": [{"message": {"content": " answer [1] "}}],
        "usage": {"prompt_tokens": total - 100, "completion_tokens": 100, "total_tokens": total},
    }
    return httpx.Response(200, json=body, headers={"x-ratelimit-remaining-tokens": "7000", **(headers or {})})


def _client(
    tmp_path: Path, responses: list[httpx.Response], limits: Limits | None = None
) -> tuple[GroqClient, FakeClock, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer test-key"
        return queue.pop(0)

    clock = FakeClock()
    http = httpx.Client(base_url="https://example.invalid", transport=httpx.MockTransport(handler))
    ledger = UsageLedger(tmp_path / "usage.json", today=lambda: "2026-11-22")
    client = GroqClient("test-key", limits or Limits(), ledger, http, sleep=clock.sleep, clock=clock)
    return client, clock, sent


MSG = [{"role": "user", "content": "hello"}]


def test_chat_success_records_usage(tmp_path: Path) -> None:
    client, _, sent = _client(tmp_path, [_ok(1200)])
    result = client.chat("m", MSG, max_tokens=256, temperature=0, reasoning_effort="none")
    assert result.content == "answer [1]" and result.total_tokens == 1200
    assert sent[0]["reasoning_effort"] == "none" and sent[0]["max_tokens"] == 256
    assert client.ledger.used("m") == (1200, 1)
    assert UsageLedger(tmp_path / "usage.json", today=lambda: "2026-11-22").used("m") == (1200, 1)


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("ragk8s.groq_client._prompt_key", lambda: "")
    with pytest.raises(GroqError, match="GROQ_API_KEY"):
        GroqClient()


def test_tokens_per_minute_pacing(tmp_path: Path) -> None:
    client, clock, _ = _client(tmp_path, [_ok(5000), _ok(5000)], Limits(tokens_per_minute=8000))
    client.chat("m", MSG, max_tokens=100)
    assert clock.slept == []
    client.chat("m", MSG, max_tokens=4000)  # 5000 used + ~4000 estimated > 8000 within the minute -> wait
    assert sum(clock.slept) >= 59


def test_429_short_waits_and_retries(tmp_path: Path) -> None:
    limited = httpx.Response(
        429, text='{"error":{"message":"Rate limit reached ... tokens per minute (TPM)"}}', headers={"retry-after": "7"}
    )
    client, clock, sent = _client(tmp_path, [limited, _ok()])
    assert client.chat("m", MSG, max_tokens=10).content == "answer [1]"
    assert clock.slept == [7.0] and len(sent) == 2


def test_429_daily_stops_and_marks_ledger(tmp_path: Path) -> None:
    limited = httpx.Response(
        429, text='{"error":{"message":"Rate limit reached ... tokens per day (TPD)"}}', headers={"retry-after": "3600"}
    )
    client, _, _ = _client(tmp_path, [limited])
    with pytest.raises(DailyQuotaReached):
        client.chat("m", MSG, max_tokens=10)
    with pytest.raises(DailyQuotaReached):  # no new request is sent
        client.chat("m", MSG, max_tokens=10)


def test_local_daily_budget_prevents_request(tmp_path: Path) -> None:
    client, _, sent = _client(tmp_path, [_ok()], Limits(tokens_per_day=2000, daily_safety_margin=0))
    client.ledger.add("m", 1900)
    with pytest.raises(DailyQuotaReached):
        client.chat("m", MSG, max_tokens=500)
    assert sent == []
    assert client.ledger.used("other-model") == (0, 0)  # budgets are per model


def test_server_error_is_retried_and_client_error_raises(tmp_path: Path) -> None:
    client, _, _ = _client(tmp_path, [httpx.Response(503, text="busy"), _ok()])
    assert client.chat("m", MSG, max_tokens=10).total_tokens == 1000
    client, _, _ = _client(tmp_path, [httpx.Response(400, text="bad model")])
    with pytest.raises(GroqError, match="400"):
        client.chat("m", MSG, max_tokens=10)


def test_cli_reports_groq_errors_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ragk8s import cli, questions_draft

    def fail(**_: Any) -> dict[str, int]:
        raise GroqError("GROQ_API_KEY is not set")

    monkeypatch.setattr(questions_draft, "draft", fail)
    assert cli.main(["questions", "draft"]) == 2
    assert "Groq error: GROQ_API_KEY is not set" in capsys.readouterr().err
