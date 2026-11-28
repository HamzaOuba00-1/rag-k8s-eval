"""Quota locks of the web app.

Three locks keep the Groq free tier (no payment method on the account) out of reach
of abuse:

1. an app-wide counter of live answers per UTC day (DailyCounter), stored in a Cloud
   Storage object so that it survives restarts, with in-memory fallback for local use;
2. a per-visitor hourly limit (VisitorLimiter). Visitors are told apart by a keyed
   hash of their IP address kept in memory only: no IP is stored or logged;
3. a pause until the next UTC midnight as soon as Groq reports a daily limit
   (Pause): the app then serves precomputed answers only (demo mode).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import httpx


def utc_now() -> datetime:
    return datetime.now(UTC)


def next_midnight(now: datetime) -> datetime:
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


class CounterStore(Protocol):
    def get(self, day: str) -> int: ...

    def try_increment(self, day: str, limit: int) -> bool:
        """Add one use for ``day`` unless ``limit`` is reached; True if added."""
        ...


class MemoryCounterStore:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def get(self, day: str) -> int:
        return self._counts.get(day, 0)

    def try_increment(self, day: str, limit: int) -> bool:
        with self._lock:
            if self._counts.get(day, 0) >= limit:
                return False
            self._counts[day] = self._counts.get(day, 0) + 1
            return True


METADATA_TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"


class GcsCounterStore:
    """One small JSON object per UTC day; updates use generation preconditions (no lost increments)."""

    def __init__(self, bucket: str, http: httpx.Client | None = None, token: Callable[[], str] | None = None) -> None:
        self.bucket = bucket
        self._http = http or httpx.Client(timeout=10.0)
        self._token = token or self._metadata_token
        self._cached: tuple[str, float] = ("", 0.0)

    def _metadata_token(self) -> str:
        value, expires = self._cached
        if value and time.monotonic() < expires - 60:
            return value
        resp = self._http.get(METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"})
        resp.raise_for_status()
        data = resp.json()
        self._cached = (data["access_token"], time.monotonic() + float(data.get("expires_in", 300)))
        return self._cached[0]

    def _name(self, day: str) -> str:
        return f"quota/{day}.json"

    def _read(self, day: str) -> tuple[int, int]:
        """(count, generation); generation 0 means the object does not exist yet."""
        url = f"https://storage.googleapis.com/storage/v1/b/{self.bucket}/o/{self._name(day).replace('/', '%2F')}"
        headers = {"Authorization": f"Bearer {self._token()}"}
        meta = self._http.get(url, headers=headers)
        if meta.status_code == 404:
            return 0, 0
        meta.raise_for_status()
        generation = int(meta.json()["generation"])
        body = self._http.get(url, params={"alt": "media", "generation": generation}, headers=headers)
        body.raise_for_status()
        return int(body.json().get("count", 0)), generation

    def get(self, day: str) -> int:
        return self._read(day)[0]

    def try_increment(self, day: str, limit: int, attempts: int = 5) -> bool:
        for _ in range(attempts):
            count, generation = self._read(day)
            if count >= limit:
                return False
            resp = self._http.post(
                f"https://storage.googleapis.com/upload/storage/v1/b/{self.bucket}/o",
                params={"uploadType": "media", "name": self._name(day), "ifGenerationMatch": generation},
                headers={"Authorization": f"Bearer {self._token()}", "Content-Type": "application/json"},
                content=json.dumps({"count": count + 1}),
            )
            if resp.status_code == 412:  # another request updated the counter first
                continue
            resp.raise_for_status()
            return True
        raise RuntimeError("quota counter: too many concurrent updates")


class VisitorLimiter:
    def __init__(
        self, hourly_limit: int, clock: Callable[[], float] = time.monotonic, key: bytes | None = None
    ) -> None:
        self.hourly_limit = hourly_limit
        self._clock = clock
        self._key = key or secrets.token_bytes(32)  # new key per process: hashes can't be linked across restarts
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def visitor_id(self, ip: str) -> str:
        return hmac.new(self._key, ip.encode(), hashlib.sha256).hexdigest()[:16]

    def _recent(self, vid: str) -> deque[float]:
        now = self._clock()
        hits = self._hits.setdefault(vid, deque())
        while hits and now - hits[0] >= 3600:
            hits.popleft()
        if not hits:
            del self._hits[vid]
            return deque()
        return hits

    def remaining(self, ip: str) -> int:
        with self._lock:
            return max(0, self.hourly_limit - len(self._recent(self.visitor_id(ip))))

    def _prune(self) -> None:
        """Forget visitors with no request in the last hour (memory stays bounded)."""
        now = self._clock()
        for vid in [v for v, hits in self._hits.items() if not hits or now - hits[-1] >= 3600]:
            del self._hits[vid]

    def try_use(self, ip: str) -> bool:
        with self._lock:
            self._calls += 1
            if self._calls % 256 == 0:
                self._prune()
            vid = self.visitor_id(ip)
            if len(self._recent(vid)) >= self.hourly_limit:
                return False
            self._hits.setdefault(vid, deque()).append(self._clock())
            return True


class Pause:
    def __init__(self, now: Callable[[], datetime] = utc_now) -> None:
        self._now = now
        self.until: datetime | None = None
        self.reason: str | None = None

    def set_until_midnight(self, reason: str) -> None:
        self.until = next_midnight(self._now())
        self.reason = reason

    @property
    def active(self) -> bool:
        if self.until and self._now() >= self.until:
            self.until, self.reason = None, None
        return self.until is not None


class QuotaGate:
    """The three locks together; ``status`` feeds the Quota schema."""

    def __init__(
        self,
        store: CounterStore,
        daily_limit: int,
        visitors: VisitorLimiter,
        live_llm: bool,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self.store = store
        self.daily_limit = daily_limit
        self.visitors = visitors
        self.live_llm = live_llm
        self.pause = Pause(now)
        self._now = now

    def day(self) -> str:
        return self._now().strftime("%Y-%m-%d")

    def status(self, ip: str) -> dict[str, Any]:
        used = self.store.get(self.day())
        reason = None
        if not self.live_llm:
            reason = "Live answers are off on this deployment."
        elif self.pause.active:
            reason = self.pause.reason
        elif used >= self.daily_limit:
            reason = "Today's live answers are used up."
        return {
            "live_enabled": reason is None,
            "remaining_today": max(0, self.daily_limit - used),
            "daily_limit": self.daily_limit,
            "visitor_remaining_this_hour": self.visitors.remaining(ip),
            "resets_at": next_midnight(self._now()).isoformat(timespec="seconds"),
            "reason": reason,
        }

    def acquire(self, ip: str) -> str | None:
        """Take one live answer; returns None when allowed, else the reason shown to the user."""
        status = self.status(ip)
        if not status["live_enabled"]:
            return str(status["reason"])
        if status["visitor_remaining_this_hour"] <= 0:
            return "You've reached the live answers for this hour."
        if not self.store.try_increment(self.day(), self.daily_limit):
            return "Today's live answers are used up."
        self.visitors.try_use(ip)
        return None
