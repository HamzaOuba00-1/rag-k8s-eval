"""HTTP hardening for the DocTruth app: security headers, CSP, request size, caching.

The Content-Security-Policy of the front end allows only this origin. The one inline
script of index.html (theme before first paint) is allowed by its SHA-256 hash,
computed from the built file at startup, so no 'unsafe-inline' is needed for scripts.
Inline *style attributes* (React `style={{...}}`) need 'unsafe-inline' for styles;
styles cannot run code.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# A question is at most 500 characters; 16 KiB leaves room for JSON and Unicode.
MAX_BODY_BYTES = 16 * 1024

COMMON_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Strict-Transport-Security": "max-age=31536000",
}

API_CSP = "default-src 'none'; frame-ancestors 'none'"

_INLINE_SCRIPT = re.compile(r"<script>(.*?)</script>", re.S)


def inline_script_hashes(html: str) -> list[str]:
    """CSP source expressions for the inline scripts (no attributes) of an HTML page."""
    return [
        "'sha256-" + base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode() + "'"
        for body in _INLINE_SCRIPT.findall(html)
    ]


def front_csp(index_html: Path | None) -> str:
    hashes = inline_script_hashes(index_html.read_text(encoding="utf-8")) if index_html and index_html.exists() else []
    script = " ".join(["'self'", *hashes])
    return (
        "default-src 'self'; "
        f"script-src {script}; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'none'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    )


def cache_control(path: str) -> str:
    if path.startswith("/api/"):
        return "no-store"
    if path.startswith("/assets/"):
        return "public, max-age=31536000, immutable"  # Vite file names carry a content hash
    if path.endswith(".png"):
        return "public, max-age=86400"
    return "no-cache"  # index.html and client-side routes: always revalidate


Handler = Callable[[Request], Awaitable[Response]]


def hardening_middleware(index_html: Path | None) -> Callable[[Request, Handler], Awaitable[Response]]:
    csp = front_csp(index_html)

    async def middleware(request: Request, call_next: Handler) -> Response:
        path = request.url.path
        if request.method in ("POST", "PUT", "PATCH"):
            length = request.headers.get("content-length")
            if length is None:
                return JSONResponse({"detail": "A Content-Length header is required."}, status_code=411)
            if not length.isdigit() or int(length) > MAX_BODY_BYTES:
                return JSONResponse({"detail": "Request body too large."}, status_code=413)
        response = await call_next(request)
        for name, value in COMMON_HEADERS.items():
            response.headers.setdefault(name, value)
        response.headers.setdefault("Content-Security-Policy", API_CSP if path.startswith("/api/") else csp)
        response.headers.setdefault("Cache-Control", cache_control(path))
        return response

    return middleware
