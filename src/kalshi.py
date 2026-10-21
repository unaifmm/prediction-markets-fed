"""
Kalshi: the "Fed decision in <month>?" markets (series KXFEDDECISION).

Each FOMC meeting is one Kalshi *event* with five *markets* (yes/no contracts):
cut more than 25 bp, cut 25 bp, hold, hike 25 bp, hike more than 25 bp.
A YES contract pays $1 if that outcome happens, so its price (in dollars) reads
as the market's probability of that outcome.

Public API, no key needed. Markets that settled before Kalshi's "historical
cutoff" (a date that moves forward over time) are served by separate
/historical/... endpoints, so every function tries the live endpoint first and
falls back to the historical one.
"""
from __future__ import annotations

import datetime as dt
import re
import urllib.parse

import pandas as pd

from .webapi import Api

MAX_CANDLES = 4800          # the API refuses requests of more than ~5,000 candles


def ts(x) -> int:
    """Anything time-like -> Unix seconds (UTC)."""
    if isinstance(x, (int, float)):
        return int(x)
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return int(t.timestamp())


def _money(d: dict | None, field: str) -> float | None:
    """A price in dollars (0 to 1). The live API sends 'close_dollars': '0.4500'; the
    historical API sends 'close': '0.4500', also in dollars (checked on the data:
    the five outcome prices of a meeting add up to about 1 only when read as dollars).
    Only a whole number (the API's old format) is in cents."""
    if not d:
        return None
    if d.get(f"{field}_dollars") is not None:
        return float(d[f"{field}_dollars"])
    v = d.get(field)
    if v is None:
        return None
    if isinstance(v, int) and not isinstance(v, bool):
        return v / 100.0
    return float(v)


def _num(m: dict, *keys) -> float | None:
    for k in keys:
        if m.get(k) not in (None, ""):
            try:
                return float(m[k])
            except (TypeError, ValueError):
                pass
    return None


# ---------------------------------------------------------------------------
# Events and markets
# ---------------------------------------------------------------------------
def list_events(api: Api, series: str) -> list[dict]:
    out, cursor = [], None
    for _ in range(50):
        code, j = api.get("/events", {"series_ticker": series, "limit": 200, "cursor": cursor})
        if code != 200:
            raise RuntimeError(f"Kalshi /events {series}: HTTP {code} {j}")
        out += j.get("events", [])
        cursor = j.get("cursor")
        if not cursor:
            break
    return out


def event_markets(api: Api, event_ticker: str) -> tuple[list[dict], str]:
    """The markets of one event, and which endpoint served them ('live' or 'historical')."""
    for path, where in (("/markets", "live"), ("/historical/markets", "historical")):
        code, j = api.get(path, {"event_ticker": event_ticker, "limit": 200})
        ms = j.get("markets", []) if code == 200 else []
        if ms:
            return ms, where
    return [], "none"


def market_row(m: dict, event: dict, where: str) -> dict:
    """The few fields we keep from a market's metadata."""
    return {
        "venue": "kalshi",
        "event": event.get("event_ticker"),
        "decision_time": event.get("strike_date"),
        "ticker": m.get("ticker"),
        "outcome": m.get("yes_sub_title") or m.get("subtitle") or m.get("title"),
        "status": m.get("status"),
        "result": m.get("result"),
        "open_time": m.get("open_time"),
        "close_time": m.get("close_time"),
        "volume": _num(m, "volume_fp", "volume"),
        "open_interest": _num(m, "open_interest_fp", "open_interest"),
        "served_by": where,
    }


# ---------------------------------------------------------------------------
# Prices (candles) and trades
# ---------------------------------------------------------------------------
def flatten_candles(cs: list[dict]) -> pd.DataFrame:
    rows = []
    for c in cs:
        r = {"ts": int(c["end_period_ts"]),
             "volume": _num(c, "volume_fp", "volume"),
             "open_interest": _num(c, "open_interest_fp", "open_interest")}
        for side in ("price", "yes_bid", "yes_ask"):
            for f in ("open", "high", "low", "close", "mean"):
                v = _money(c.get(side), f)
                if v is not None or f != "mean":
                    r[f"{side}_{f}"] = v
        rows.append(r)
    return pd.DataFrame(rows)


def candles(api: Api, series: str, ticker: str, start, end, interval: int,
            historical: bool = False) -> pd.DataFrame:
    """Candles of `interval` minutes (1, 60 or 1440) between start and end, in as
    many requests as needed. Columns: ts (end of the candle, Unix UTC), last-trade
    price OHLC (+ mean), best YES bid and ask OHLC, volume, open interest.
    `historical`: the market settled before Kalshi's historical cutoff."""
    a, b = ts(start), ts(end)
    step = MAX_CANDLES * interval * 60
    q = lambda x: urllib.parse.quote(str(x), safe="")      # old tickers contain '>' (e.g. H>25)
    paths = [f"/series/{q(series)}/markets/{q(ticker)}/candlesticks", f"/historical/markets/{q(ticker)}/candlesticks"]
    if historical:
        paths.reverse()
    parts = []
    while a < b:
        e = min(a + step, b)
        p = {"start_ts": a, "end_ts": e, "period_interval": interval}
        for path in paths:                       # the other endpoint only if this one refuses
            code, j = api.get(path, p)
            if code == 200:
                if j.get("candlesticks"):
                    parts.append(flatten_candles(j["candlesticks"]))
                break
        a = e
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).drop_duplicates("ts").sort_values("ts")
    df.insert(0, "ticker", ticker)
    return df.reset_index(drop=True)


def flatten_trades(tr: list[dict]) -> pd.DataFrame:
    rows = []
    for t in tr:
        yes = t.get("yes_price_dollars")
        yes = float(yes) if yes is not None else (float(t["yes_price"]) / 100 if t.get("yes_price") is not None else None)
        rows.append({"time": t.get("created_time"), "ticker": t.get("ticker"), "yes_price": yes,
                     "count": _num(t, "count_fp", "count"), "taker_side": t.get("taker_side"),
                     "is_block_trade": t.get("is_block_trade"), "trade_id": t.get("trade_id")})
    df = pd.DataFrame(rows)
    if len(df):
        df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed")
        df = df.drop_duplicates("trade_id").sort_values("time").reset_index(drop=True)
    return df


def trades(api: Api, ticker: str, start, end, historical: bool = False,
           max_pages: int = 100) -> pd.DataFrame:
    """Every trade in one market between start and end (exact timestamps).
    `historical`: the window ends before Kalshi's historical cutoff for trades."""
    a, b = ts(start), ts(end)
    paths = ["/markets/trades", "/historical/trades"]
    if historical:
        paths.reverse()
    for path in paths:                           # the other endpoint only if this one refuses
        out, cursor, refused = [], None, False
        for _ in range(max_pages):
            code, j = api.get(path, {"ticker": ticker, "min_ts": a, "max_ts": b, "limit": 1000, "cursor": cursor})
            if code != 200:
                refused = True
                break
            out += j.get("trades", [])
            cursor = j.get("cursor")
            if not cursor:
                break
        if not refused:
            return flatten_trades(out)
    return pd.DataFrame()


def cutoffs(api: Api) -> dict:
    """Kalshi's historical cutoffs as Unix seconds (markets settled / trades created before them
    live under /historical/...). Missing -> 0, i.e. treat everything as live."""
    code, j = api.get("/historical/cutoff")
    j = j if code == 200 else {}
    return {k: ts(v) if v else 0 for k, v in j.items() if k.endswith("_ts")} | \
        {k: 0 for k in ("market_settled_ts", "trades_created_ts") if k not in j}


def file_name(ticker: str) -> str:
    """A ticker made safe for file names on every system: '>' and '<' are not allowed
    on Windows, so FEDDECISION-23JUN-H>25 is saved as FEDDECISION-23JUN-Hgt25."""
    t = str(ticker).replace(">", "gt").replace("<", "lt")
    return re.sub(r"[^A-Za-z0-9._-]", "_", t)


def now_utc() -> int:
    return int(dt.datetime.now(dt.timezone.utc).timestamp())
