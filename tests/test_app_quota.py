from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

pytest.importorskip("fastapi")

from app.quota import GcsCounterStore, MemoryCounterStore, Pause, QuotaGate, VisitorLimiter


class FakeGcs:
    """Objects with generations; a write with a stale ifGenerationMatch gets HTTP 412."""

    def __init__(self, race_once: bool = False) -> None:
        self.objects: dict[str, tuple[int, bytes]] = {}
        self.race_once = race_once

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            name = request.url.path.rsplit("/o/", 1)[1].replace("%2F", "/")
            if name not in self.objects:
                return httpx.Response(404)
            generation, body = self.objects[name]
            if request.url.params.get("alt") == "media":
                return httpx.Response(200, content=body)
            return httpx.Response(200, json={"generation": str(generation)})
        name = request.url.params["name"]
        expected = int(request.url.params["ifGenerationMatch"])
        current = self.objects.get(name, (0, b""))[0]
        if self.race_once:  # someone else wrote between our read and our write
            self.race_once = False
            self.objects[name] = (current + 1, json.dumps({"count": 1}).encode())
            return httpx.Response(412)
        if expected != current:
            return httpx.Response(412)
        self.objects[name] = (current + 1, request.content)
        return httpx.Response(200, json={})


def _store(fake: FakeGcs) -> GcsCounterStore:
    return GcsCounterStore("bucket", http=httpx.Client(transport=httpx.MockTransport(fake.handler)), token=lambda: "t")


def test_gcs_counter_counts_up_to_the_limit() -> None:
    store = _store(FakeGcs())
    assert [store.try_increment("2026-11-23", 2) for _ in range(3)] == [True, True, False]
    assert store.get("2026-11-23") == 2 and store.get("2026-11-24") == 0


def test_gcs_counter_retries_after_a_concurrent_write() -> None:
    fake = FakeGcs(race_once=True)
    store = _store(fake)
    assert store.try_increment("2026-11-23", 5) is True
    assert store.get("2026-11-23") == 2  # the concurrent increment is kept


def test_pause_lifts_at_utc_midnight() -> None:
    now = [datetime(2026, 11, 23, 22, 0, tzinfo=UTC)]
    pause = Pause(lambda: now[0])
    pause.set_until_midnight("limit")
    assert pause.active
    now[0] = datetime(2026, 11, 24, 0, 0, 1, tzinfo=UTC)
    assert not pause.active and pause.reason is None


def test_visitor_window_slides_over_one_hour() -> None:
    t = [0.0]
    limiter = VisitorLimiter(2, clock=lambda: t[0], key=b"k")
    assert limiter.try_use("ip") and limiter.try_use("ip") and not limiter.try_use("ip")
    t[0] = 3600.0
    assert limiter.remaining("ip") == 2


def test_gate_counts_only_allowed_answers() -> None:
    gate = QuotaGate(MemoryCounterStore(), 1, VisitorLimiter(5), live_llm=True)
    assert gate.acquire("a") is None
    assert gate.acquire("b") == "Today's live answers are used up."
    assert gate.status("b")["remaining_today"] == 0


def test_visitor_limiter_forgets_idle_visitors() -> None:
    t = [0.0]
    limiter = VisitorLimiter(5, clock=lambda: t[0], key=b"k")
    for i in range(255):
        limiter.try_use(f"10.0.0.{i}")
    assert len(limiter._hits) == 255
    t[0] = 3601.0
    limiter.try_use("10.0.1.1")  # 256th call: prunes the visitors idle for an hour
    assert len(limiter._hits) == 1
