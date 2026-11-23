"""Runtime settings of the web app, read from environment variables only."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DOCTRUTH_DATA_DIR", "app_data")))
    static_dir: Path = field(default_factory=lambda: Path(os.environ.get("DOCTRUTH_STATIC_DIR", "web/dist")))
    # Live answers for the whole app per UTC day. Groq free tier: 200k tokens/day for
    # the model; one answer used ~1.4k tokens in the evaluation runs, so 80 answers
    # keep a wide margin (estimate, not a measured app figure).
    daily_limit: int = field(default_factory=lambda: _int("DOCTRUTH_DAILY_LIMIT", 80))
    visitor_hourly_limit: int = field(default_factory=lambda: _int("DOCTRUTH_VISITOR_HOURLY", 5))
    quota_bucket: str = field(default_factory=lambda: os.environ.get("DOCTRUTH_QUOTA_BUCKET", "").strip())
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", "").strip(), repr=False)
    threads: int = field(default_factory=lambda: _int("DOCTRUTH_THREADS", 1))

    @property
    def live_llm(self) -> bool:
        return bool(self.groq_api_key)
