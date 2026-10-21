"""
Polymarket: the monthly "Fed decision in <month>?" events.

Each FOMC meeting is one Polymarket *event* with four or five *markets*
("No change", "25 bps decrease", ...). Each market has a YES and a NO token;
the YES price (0 to 1) reads as the probability of that outcome.

Three public APIs, no key needed:
  * Gamma  - which events and markets exist;
  * CLOB   - price history of a token (`prices-history`);
  * Data   - trades, with exact timestamps.
Trades come with the trader's public wallet and profile name; we do not keep
those fields, only what the research needs.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from .kalshi import ts
from .webapi import Api

TRADE_FIELDS = ["timestamp", "conditionId", "asset", "outcome", "outcomeIndex", "side", "price", "size"]


# ---------------------------------------------------------------------------
# Events and markets
# ---------------------------------------------------------------------------
def search_events(gamma: Api, queries, title_pattern: str) -> list[dict]:
    """All events whose title looks like a monthly Fed decision market."""
    found = {}
    pat = re.compile(title_pattern, re.IGNORECASE)
    for q in queries:
        for extra in ({}, {"events_status": "closed"}, {"keep_closed_markets": 1}):
            for page in range(1, 6):
                code, j = gamma.get("/public-search", {"q": q, "limit_per_type": 50, "page": page, **extra})
                if code != 200:
                    break
                for e in j.get("events") or []:
                    if pat.match((e.get("title") or "").strip()):
                        found[str(e.get("id"))] = e
                if not (j.get("pagination") or {}).get("hasMore"):
                    break
    return sorted(found.values(), key=lambda e: e.get("endDate") or "")


def event_markets(gamma: Api, event: dict) -> list[dict]:
    """Markets of an event (search results sometimes leave them out)."""
    ms = event.get("markets") or []
    if not ms:
        code, j = gamma.get("/events", {"slug": event.get("slug")})
        if code == 200 and isinstance(j, list) and j:
            ms = j[0].get("markets") or []
    return ms


def market_row(m: dict, event: dict) -> dict:
    tokens = json.loads(m.get("clobTokenIds") or "[]")
    return {
        "venue": "polymarket",
        "event": event.get("slug"),
        "event_id": str(event.get("id")),
        "event_end": event.get("endDate"),
        "market_id": str(m.get("id")),
        "condition_id": m.get("conditionId"),
        "yes_token": tokens[0] if tokens else None,
        "outcome": m.get("groupItemTitle") or m.get("question"),
        "question": m.get("question"),
        "closed": m.get("closed"),
        "final_prices": m.get("outcomePrices"),
        "start_time": m.get("startDate") or event.get("startDate"),
        "closed_time": m.get("closedTime") or event.get("closedTime"),
        "volume": float(m.get("volumeNum") or m.get("volume") or 0),
        "tick_size": m.get("orderPriceMinTickSize"),
        "fee_schedule": json.dumps(m.get("feeSchedule")) if m.get("feeSchedule") else None,
    }


# ---------------------------------------------------------------------------
# Prices and trades
# ---------------------------------------------------------------------------
def price_history(clob: Api, token: str, start, end, fidelity: int, chunk_days: float) -> pd.DataFrame:
    """YES-token price every `fidelity` minutes between start and end, fetched in
    chunks (the API returns nothing for very long ranges on resolved markets)."""
    a, b = ts(start), ts(end)
    step = int(chunk_days * 86400)
    parts = []
    while a < b:
        e = min(a + step, b)
        code, j = clob.get("/prices-history", {"market": token, "startTs": a, "endTs": e, "fidelity": fidelity})
        h = j.get("history", []) if code == 200 else []
        if h:
            parts.append(pd.DataFrame(h))
        a = e
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).rename(columns={"t": "ts"}).drop_duplicates("ts").sort_values("ts")
    df.insert(0, "yes_token", token)
    return df.reset_index(drop=True)


def event_trades(data: Api, event_id: str, start, end, depth: int = 0) -> pd.DataFrame:
    """Every trade in an event's markets between start and end. The API pages
    with an offset of at most 10,000, so a busy window is split in two."""
    a, b = ts(start), ts(end)
    rows = []
    for offset in (0, 10000):
        code, j = data.get("/trades", {"eventId": event_id, "start": a, "end": b,
                                       "limit": 10000, "offset": offset})   # taker side only: each trade once
        got = j if code == 200 and isinstance(j, list) else []
        rows += got
        if len(got) < 10000:
            break
    else:
        if depth < 6 and b - a > 60:                 # still full: split the window
            mid = (a + b) // 2
            return pd.concat([event_trades(data, event_id, a, mid, depth + 1),
                              event_trades(data, event_id, mid, b, depth + 1)], ignore_index=True)
    if not rows:
        return pd.DataFrame(columns=TRADE_FIELDS)
    df = pd.DataFrame(rows)
    keep = [c for c in TRADE_FIELDS + ["transactionHash"] if c in df.columns]
    df = df[keep].drop_duplicates()                  # the hash only tells repeated rows apart
    df = df.drop(columns=["transactionHash"], errors="ignore")
    return df.sort_values("timestamp").reset_index(drop=True)
