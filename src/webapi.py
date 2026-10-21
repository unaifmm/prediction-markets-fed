"""
A small, polite JSON client for public APIs (Kalshi, Polymarket, FRED).

* waits a little between calls so we stay well under the sites' rate limits;
* retries network errors, rate limits (HTTP 429) and server errors (5xx),
  waiting longer each time;
* never raises on an HTTP error: it returns (status, body) so the caller can
  decide (e.g. "not found here, try the historical endpoint").
Only the Python standard library is used.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request


class Api:
    def __init__(self, base: str, pause: float = 0.12, timeout: float = 40.0):
        self.base = base.rstrip("/")
        self.pause = pause
        self.timeout = timeout
        self.calls = 0
        self._last = 0.0

    def _wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.pause:
            time.sleep(self.pause - gap)
        self._last = time.monotonic()

    def get(self, path: str, params: dict | None = None, tries: int = 5, raw: bool = False):
        """GET base+path?params. Returns (status or None, parsed JSON | text | {'error': ...})."""
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        for attempt in range(tries):
            self._wait()
            self.calls += 1
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "research-download/1.0",
                                                           "Accept": "application/json, text/csv"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = r.read().decode("utf-8", errors="replace")
                    return r.status, (body if raw else json.loads(body))
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", errors="replace")[:300]
                if (e.code == 429 or e.code >= 500) and attempt < tries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return e.code, {"error": text}
            except Exception as e:                       # timeouts, dropped connections, bad JSON
                if attempt < tries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None, {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        return None, {"error": "gave up"}
