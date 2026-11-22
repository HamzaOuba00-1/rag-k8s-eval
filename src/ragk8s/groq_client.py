"""Minimal Groq client (OpenAI-compatible chat completions) built for a free-tier quota.

Free-tier limits are per model: requests and tokens per minute (RPM/TPM) and per
day (RPD/TPD). This client:

- paces requests so the tokens sent in any 60 s window stay under the TPM limit,
  and also waits when the ``x-ratelimit-remaining-tokens`` header says so;
- keeps a local ledger of tokens used per model and UTC day
  (.cache/groq_usage.json) and refuses to start a request that would cross the
  daily budget (raises DailyQuotaReached, so a run can stop cleanly and resume
  the next day from its cache);
- on HTTP 429, waits ``retry-after`` seconds, or stops for the day when the
  limit hit is a daily one;
- retries transient network and 5xx errors a few times.

The API key is read from the GROQ_API_KEY environment variable and is never
written anywhere.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from . import config

BASE_URL = "https://api.groq.com/openai/v1"
USAGE_PATH = config.CACHE_DIR / "groq_usage.json"


class DailyQuotaReached(RuntimeError):
    """The daily token or request budget of a model is used up; resume tomorrow."""


class GroqError(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    tokens_per_minute: int = 8_000
    tokens_per_day: int = 200_000
    requests_per_minute: int = 30
    requests_per_day: int = 1_000
    daily_safety_margin: int = 5_000
    """Tokens kept in reserve so that an estimate error never hits the hard limit."""


FREE_TIER = Limits()


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    seconds: float
    rate_limit: dict[str, str] = field(default_factory=dict)


_DURATION_RE = re.compile(
    r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$"
)


def parse_duration(value: str | None) -> float:
    """Groq reset headers: ``"7.66s"``, ``"2m59.56s"``, ``"1h2m3s"``, ``"120ms"`` -> seconds."""
    if not value:
        return 0.0
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    match = _DURATION_RE.fullmatch(value)
    if not match or not any(match.groups()):
        return 0.0
    h, m, s, ms = (float(g) if g else 0.0 for g in match.groups())
    return h * 3600 + m * 60 + s + ms / 1000


def estimate_tokens(messages: Sequence[Mapping[str, str]], max_tokens: int) -> int:
    """Conservative upper bound used for pacing: ~3.5 chars per token plus the completion cap."""
    chars = sum(len(m.get("content", "")) for m in messages)
    return int(chars / 3.5) + 16 * len(messages) + max_tokens


def utc_day() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


class UsageLedger:
    """Tokens and requests per model and UTC day, persisted between runs."""

    def __init__(self, path: Path = USAGE_PATH, today: Callable[[], str] = utc_day) -> None:
        self.path = path
        self._today = today
        self._data: dict[str, dict[str, dict[str, int]]] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )

    def used(self, model: str) -> tuple[int, int]:
        day = self._data.get(model, {}).get(self._today(), {})
        return day.get("tokens", 0), day.get("requests", 0)

    def add(self, model: str, tokens: int, requests: int = 1) -> None:
        day = self._data.setdefault(model, {}).setdefault(self._today(), {"tokens": 0, "requests": 0})
        day["tokens"] += tokens
        day["requests"] += requests
        self._save()

    def mark_exhausted(self, model: str, limits: Limits) -> None:
        day = self._data.setdefault(model, {}).setdefault(self._today(), {"tokens": 0, "requests": 0})
        day["tokens"] = max(day["tokens"], limits.tokens_per_day)
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


class GroqClient:
    def __init__(
        self,
        api_key: str | None = None,
        limits: Limits = FREE_TIER,
        ledger: UsageLedger | None = None,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        max_retries: int = 5,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise GroqError("GROQ_API_KEY is not set (see docs/REPRODUCE.md)")
        self.limits = limits
        self.ledger = ledger or UsageLedger()
        self._http = http or httpx.Client(base_url=BASE_URL, timeout=120.0)
        self._headers = {"Authorization": f"Bearer {key}"}
        self._sleep = sleep
        self._clock = clock
        self._max_retries = max_retries
        self._window: deque[tuple[float, int]] = deque()  # (time, tokens) of the last 60 s
        self._requests: deque[float] = deque()
        self._wait_until = 0.0

    # --- pacing --------------------------------------------------------------------

    def _check_daily(self, model: str, estimate: int) -> None:
        tokens, requests = self.ledger.used(model)
        if tokens + estimate > self.limits.tokens_per_day - self.limits.daily_safety_margin:
            raise DailyQuotaReached(
                f"{model}: {tokens:,} tokens used today (UTC), daily budget {self.limits.tokens_per_day:,}. "
                "Rerun tomorrow: finished items are cached."
            )
        if requests + 1 > self.limits.requests_per_day:
            raise DailyQuotaReached(f"{model}: {requests} requests used today (UTC). Rerun tomorrow.")

    def _pace(self, estimate: int) -> None:
        now = self._clock()
        if self._wait_until > now:
            self._sleep(self._wait_until - now)
        while True:
            now = self._clock()
            while self._window and now - self._window[0][0] >= 60:
                self._window.popleft()
            while self._requests and now - self._requests[0] >= 60:
                self._requests.popleft()
            used = sum(t for _, t in self._window)
            tokens_ok = not self._window or used + estimate <= self.limits.tokens_per_minute
            requests_ok = len(self._requests) < self.limits.requests_per_minute
            if tokens_ok and requests_ok:
                return
            blocking = ([self._window[0][0]] if not tokens_ok else []) + (
                [self._requests[0]] if not requests_ok else []
            )
            self._sleep(max(0.5, 60 - (now - min(blocking)) + 0.25))

    # --- request -------------------------------------------------------------------

    def chat(self, model: str, messages: Sequence[Mapping[str, str]], max_tokens: int, **params: Any) -> ChatResult:
        estimate = estimate_tokens(messages, max_tokens)
        payload = {"model": model, "messages": list(messages), "max_tokens": max_tokens, **params}
        for attempt in range(self._max_retries + 1):
            self._check_daily(model, estimate)
            self._pace(estimate)
            started = self._clock()
            try:
                resp = self._http.post("/chat/completions", json=payload, headers=self._headers)
            except httpx.TransportError as exc:
                if attempt == self._max_retries:
                    raise GroqError(f"network error: {exc}") from exc
                self._sleep(5.0 * (attempt + 1))
                continue
            rate = {k: v for k, v in resp.headers.items() if k.lower().startswith("x-ratelimit") or k == "retry-after"}
            if resp.status_code == 429:
                self._handle_429(model, resp, rate)
                continue
            if resp.status_code >= 500:
                if attempt == self._max_retries:
                    raise GroqError(f"server error {resp.status_code}: {resp.text[:300]}")
                self._sleep(5.0 * (attempt + 1))
                continue
            if resp.status_code != 200:
                raise GroqError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            return self._success(model, resp.json(), rate, started)
        raise GroqError(f"rate limited {self._max_retries + 1} times in a row")

    def _handle_429(self, model: str, resp: httpx.Response, rate: Mapping[str, str]) -> None:
        retry_after = parse_duration(rate.get("retry-after"))
        message = resp.text.lower()
        daily = "per day" in message or "tpd" in message or "rpd" in message or retry_after > 600
        if daily:
            self.ledger.mark_exhausted(model, self.limits)
            raise DailyQuotaReached(
                f"{model}: daily limit reached (HTTP 429). Rerun tomorrow: finished items are cached."
            )
        self._sleep(max(retry_after, 2.0))

    def _success(self, model: str, data: dict[str, Any], rate: dict[str, str], started: float) -> ChatResult:
        usage = data.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or prompt + completion)
        now = self._clock()
        self._window.append((now, total))
        self._requests.append(now)
        self.ledger.add(model, total)
        remaining = rate.get("x-ratelimit-remaining-tokens")
        if remaining is not None and remaining.isdigit() and int(remaining) < self.limits.tokens_per_minute // 4:
            self._wait_until = now + parse_duration(rate.get("x-ratelimit-reset-tokens"))
        message = (data.get("choices") or [{}])[0].get("message") or {}
        return ChatResult(
            content=str(message.get("content") or "").strip(),
            model=str(data.get("model", model)),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            seconds=round(now - started, 2),
            rate_limit=rate,
        )
