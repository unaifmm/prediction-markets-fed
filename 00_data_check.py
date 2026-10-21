"""
STEP 0 - Data check: what data exists? (read-only, about 1-2 minutes)

A one-off check run on 20 Oct 2026 before designing the study. Its findings are
summarised in the README; it is kept to show how the data sources were chosen.

Before designing anything we check what data actually exists:
  A) Kalshi       - Fed markets: how many meetings, 1-minute price history, trades
  B) Polymarket   - Fed decision markets: same questions
  C) LSE          - US Treasury yields and the effective fed funds rate:
                    how fine is the intraday data around an FOMC decision?

Run:  python 00_data_check.py

It only READS public market data (no account, nothing is traded) and writes
    data_check/report.txt   <- the summary
    data_check/*.json       <- small raw samples
The London Strategic Edge key is read from ~/.lse/config.json and is
never printed or saved.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(errors="replace")

HERE = Path(__file__).resolve().parent
OUT = HERE / "data_check"
OUT.mkdir(exist_ok=True)
LOG: list[str] = []

# The FOMC decision we use as the test case: 16 Sep 2026, 14:00 New York = 18:00 UTC
WIN_START, WIN_END = "2026-09-14", "2026-09-18"


def say(*parts) -> None:
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    LOG.append(line)


def unix(day: str) -> int:
    return int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp())


def utc(t) -> str:
    try:
        return dt.datetime.fromtimestamp(int(t), dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(t)


def get(url: str, params: dict | None = None, tries: int = 3):
    """GET a JSON endpoint. Returns (http_status or None, parsed json or {'error': ...})."""
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-data-check/1.0",
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:200]
            if e.code == 429 and i < tries - 1:
                time.sleep(3 * (i + 1))
                continue
            return e.code, {"error": body}
        except Exception as e:  # network problems, bad JSON, ...
            if i < tries - 1:
                time.sleep(2)
                continue
            return None, {"error": str(e)[:200]}


def save(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=1, default=str)[:3_000_000], encoding="utf-8")


def spacing(times: list[int]) -> str:
    """Median gap between consecutive points, in minutes."""
    t = sorted(int(x) for x in times)
    if len(t) < 2:
        return "n/a"
    gaps = sorted(b - a for a, b in zip(t, t[1:]))
    return f"{gaps[len(gaps) // 2] / 60:.1f} min"


# ============================================================================
# A) KALSHI
# ============================================================================
def kalshi() -> None:
    say("\n" + "=" * 70 + "\nA) KALSHI\n" + "=" * 70)
    base = None
    for b in ("https://api.elections.kalshi.com/trade-api/v2", "https://external-api.kalshi.com/trade-api/v2"):
        code, j = get(b + "/series/KXFED")
        say(f"connect {b}: HTTP {code}")
        if code == 200:
            base = b
            save("kalshi_series_KXFED.json", j)
            s = j.get("series", j)
            say(f"  series KXFED: {s.get('title')} | frequency: {s.get('frequency')}")
            break
    if not base:
        say("!! Kalshi not reachable from this computer.")
        return

    code, j = get(base + "/historical/cutoff")
    say(f"historical cutoff: HTTP {code} {json.dumps(j)[:200]}")

    events_by_series = {}
    for series in ("KXFED", "FED", "KXFEDDECISION", "FEDDECISION"):
        evs, cursor = [], None
        for _ in range(25):
            p = {"series_ticker": series, "limit": 200}
            if cursor:
                p["cursor"] = cursor
            code, j = get(base + "/events", p)
            if code != 200:
                say(f"  events {series}: HTTP {code} {str(j.get('error', ''))[:120]}")
                break
            evs += j.get("events", [])
            cursor = j.get("cursor")
            if not cursor:
                break
        tick = sorted(e.get("event_ticker", "") for e in evs)
        events_by_series[series] = evs
        say(f"series {series}: {len(evs)} events" + (f" | first {tick[0]} | last {tick[-1]}" if tick else ""))
        if tick:
            say("   all: " + ", ".join(tick))
    save("kalshi_events.json", events_by_series)

    def markets_of(event: str) -> list[dict]:
        for path in ("/markets", "/historical/markets"):
            code, j = get(base + path, {"event_ticker": event, "limit": 200})
            ms = j.get("markets", []) if code == 200 else []
            say(f"   markets of {event} via {path}: HTTP {code}, {len(ms)} markets")
            if ms:
                return ms
        return []

    def candles(series: str, ticker: str, start: int, end: int, interval: int) -> list[dict]:
        p = {"start_ts": start, "end_ts": end, "period_interval": interval}
        for path in (f"/series/{series}/markets/{ticker}/candlesticks",
                     f"/historical/markets/{ticker}/candlesticks"):
            code, j = get(base + path, p)
            cs = j.get("candlesticks", []) if code == 200 else []
            say(f"   candles {interval}-min via {path.split('/markets')[0] or '/'}: HTTP {code}, {len(cs)} candles"
                + ("" if code == 200 else f" {str(j.get('error', ''))[:100]}"))
            if cs:
                return cs
        return []

    def trades(ticker: str, start: int, end: int) -> list[dict]:
        for path in ("/markets/trades", "/historical/trades"):
            out, cursor = [], None
            for _ in range(10):
                p = {"ticker": ticker, "min_ts": start, "max_ts": end, "limit": 1000}
                if cursor:
                    p["cursor"] = cursor
                code, j = get(base + path, p)
                if code != 200:
                    say(f"   trades via {path}: HTTP {code} {str(j.get('error', ''))[:100]}")
                    break
                out += j.get("trades", [])
                cursor = j.get("cursor")
                if not cursor:
                    break
            say(f"   trades via {path}: {len(out)} trades")
            if out:
                return out
        return []

    def test_event(series: str, event: str, tag: str) -> None:
        say(f"\n-- test event {event} ({tag})")
        ms = markets_of(event)
        if not ms:
            return
        for m in ms:
            say(f"   {m.get('ticker')} | {m.get('yes_sub_title') or m.get('subtitle') or ''} | "
                f"volume {m.get('volume')} | result {m.get('result')} | close {m.get('close_time')}")
        top = max(ms, key=lambda m: float(m.get("volume") or 0))
        close = top.get("close_time") or f"{WIN_END}T00:00:00Z"
        end = int(dt.datetime.fromisoformat(close.replace("Z", "+00:00")).timestamp())
        start = end - 4 * 86400
        say(f"   most traded: {top.get('ticker')}  window {utc(start)} -> {utc(end)} UTC")
        cs = candles(series, top["ticker"], start, end, 1)
        if cs:
            save(f"kalshi_candles_1m_{top['ticker']}.json", cs)
            say(f"   first candle keys: {list(cs[0].keys())}")
            say(f"   first {utc(cs[0].get('end_period_ts'))}, last {utc(cs[-1].get('end_period_ts'))}, "
                f"spacing {spacing([c.get('end_period_ts', 0) for c in cs])}")
            with_vol = sum(1 for c in cs if float(c.get("volume") or 0) > 0)
            say(f"   minutes with at least one trade: {with_vol} of {len(cs)}")
        else:
            cs60 = candles(series, top["ticker"], end - 60 * 86400, end, 60)
            if cs60:
                save(f"kalshi_candles_60m_{top['ticker']}.json", cs60)
        tr = trades(top["ticker"], start, end)
        if tr:
            save(f"kalshi_trades_{top['ticker']}.json", tr[:2000])
            say(f"   trade keys: {list(tr[0].keys())}")

    # the September 2026 meeting, then the oldest meeting we can find
    for series in ("KXFED", "KXFEDDECISION"):
        evs = [e["event_ticker"] for e in events_by_series.get(series, []) if "26SEP" in e.get("event_ticker", "")]
        if evs:
            test_event(series, evs[0], "Sep 2026 meeting")
    for series in ("FED", "KXFED"):
        evs = sorted(e["event_ticker"] for e in events_by_series.get(series, []))
        if evs:
            test_event(series, evs[0], "oldest meeting in this series")
            break


# ============================================================================
# B) POLYMARKET
# ============================================================================
def polymarket() -> None:
    say("\n" + "=" * 70 + "\nB) POLYMARKET\n" + "=" * 70)
    G, C = "https://gamma-api.polymarket.com", "https://clob.polymarket.com"
    code, _ = get(C + "/time")
    say(f"connect CLOB: HTTP {code}")

    events = {}
    for q in ("Fed decision", "Fed interest rates", "FOMC"):
        for extra in ({}, {"events_status": "closed"}, {"keep_closed_markets": 1}):
            for page in (1, 2, 3):
                code, j = get(G + "/public-search", {"q": q, "limit_per_type": 50, "page": page, **extra})
                if code != 200:
                    say(f"  search '{q}' {extra} p{page}: HTTP {code} {str(j.get('error', ''))[:100]}")
                    break
                got = j.get("events") or []
                for e in got:
                    t = (e.get("title") or "").lower()
                    if "fed" in t and ("decision" in t or "rate" in t):
                        events[e.get("slug")] = e
                if not (j.get("pagination") or {}).get("hasMore"):
                    break
    evs = sorted(events.values(), key=lambda e: e.get("endDate") or "")
    say(f"Fed decision/rate events found: {len(evs)}")
    for e in evs:
        say(f"   {str(e.get('endDate'))[:10]} | {e.get('title')} | slug {e.get('slug')} | "
            f"volume {round(float(e.get('volume') or 0)):,} | markets {len(e.get('markets') or [])} | closed {e.get('closed')}")
    save("polymarket_events.json", evs)
    if not evs:
        return

    def test(e: dict, tag: str) -> None:
        say(f"\n-- test event {e.get('slug')} ({tag})")
        ms = e.get("markets") or []
        if not ms:
            code, j = get(G + "/events", {"slug": e.get("slug")})
            ms = (j[0].get("markets") if code == 200 and isinstance(j, list) and j else []) or []
        for m in ms:
            say(f"   {m.get('groupItemTitle') or m.get('question')} | volume {round(float(m.get('volumeNum') or m.get('volume') or 0)):,}"
                f" | prices {m.get('outcomePrices')}")
        if not ms:
            return
        top = max(ms, key=lambda m: float(m.get("volumeNum") or m.get("volume") or 0))
        tokens = json.loads(top.get("clobTokenIds") or "[]")
        if not tokens:
            say("   no CLOB token ids")
            return
        end_s = (top.get("endDate") or e.get("endDate") or f"{WIN_END}T00:00:00Z").replace("Z", "+00:00")
        end = int(dt.datetime.fromisoformat(end_s).timestamp())
        say(f"   most traded: {top.get('groupItemTitle') or top.get('question')} | YES token ...{tokens[0][-8:]}")
        tries = [("1-min, last 4 days before end", {"startTs": end - 4 * 86400, "endTs": end, "fidelity": 1}),
                 ("1-min, 'max' interval", {"interval": "max", "fidelity": 1}),
                 ("1-hour, 'max' interval", {"interval": "max", "fidelity": 60})]
        for label, p in tries:
            code, j = get(C + "/prices-history", {"market": tokens[0], **p})
            h = j.get("history", []) if code == 200 else []
            say(f"   prices-history {label}: HTTP {code}, {len(h)} points"
                + (f", {utc(h[0]['t'])} -> {utc(h[-1]['t'])}, spacing {spacing([x['t'] for x in h])}" if h else
                   ("" if code == 200 else f" {str(j.get('error', ''))[:100]}")))
            if h:
                save(f"polymarket_history_{label.split(',')[0]}_{e.get('slug')}.json"[:120], h)

    sep = [e for e in evs if str(e.get("endDate", "")).startswith("2026-09")]
    test(sep[0] if sep else evs[-1], "Sep 2026 meeting" if sep else "most recent")
    test(evs[0], "oldest event found")


# ============================================================================
# C) LONDON STRATEGIC EDGE (rates side)
# ============================================================================
def lse() -> None:
    say("\n" + "=" * 70 + "\nC) LONDON STRATEGIC EDGE - rates\n" + "=" * 70)
    sys.path.insert(0, str(HERE))
    try:
        from src.lse import connect
        client = connect()
    except BaseException as exc:
        say(f"!! could not connect to LSE: {str(exc)[:150]}")
        return
    tests = [("USGB2Y", "sovereign_yields"), ("USGB3M", "sovereign_yields"),
             ("USGB1Y", "sovereign_yields"), ("USGB10Y", "sovereign_yields"), ("usaeffr", "economics")]
    for sym, ds in tests:
        done = False
        for tf in ("1m", "5m", "1h", "1d"):
            try:
                rows = client.candles(sym, tf, start=WIN_START, end=WIN_END, limit=5000, dataset=ds)
            except Exception as exc:
                say(f"   {sym} {tf}: error {str(exc)[:120]}")
                continue
            rows = list(rows or [])
            if not rows:
                say(f"   {sym} {tf}: 0 rows")
                continue
            key = "timestamp" if "timestamp" in rows[0] else "ts"
            times = [r.get(key) for r in rows]
            say(f"   {sym} {tf}: {len(rows)} rows, {times[0]} -> {times[-1]}, keys {list(rows[0].keys())}")
            around = [r for r in rows if str(r.get(key, "")).startswith("2026-09-16T17") or
                      str(r.get(key, "")).startswith("2026-09-16T18") or
                      str(r.get(key, "")).startswith("2026-09-16 17") or
                      str(r.get(key, "")).startswith("2026-09-16 18")]
            say(f"      rows between 17:00 and 18:59 UTC on 16 Sep (decision at 18:00 UTC): {len(around)}")
            for r in around[:8]:
                say(f"      {r}")
            save(f"lse_{sym}_{tf}.json", rows[:3000])
            done = True
            break
        if not done:
            say(f"   {sym}: no data in any timeframe")
    # how far back does the daily 2-year yield go?
    try:
        rows = list(client.candles("USGB2Y", "1d", start="2016-01-01", end="2016-02-01", limit=100,
                                   dataset="sovereign_yields") or [])
        say(f"   USGB2Y daily in Jan 2016: {len(rows)} rows")
    except Exception as exc:
        say(f"   USGB2Y daily 2016: error {str(exc)[:120]}")


if __name__ == "__main__":
    say(f"Data check run at {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC")
    try:
        for part in (kalshi, polymarket, lse):
            try:
                part()
            except Exception as exc:
                say(f"!! {part.__name__} stopped: {type(exc).__name__}: {str(exc)[:200]}")
    finally:
        (OUT / "report.txt").write_text("\n".join(LOG), encoding="utf-8")
        print(f"\nDone. Summary saved to {OUT / 'report.txt'} - tell Claude it has finished.")
