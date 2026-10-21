"""
London Strategic Edge (LSE) data: download and load 1-minute bars, 1-second
bars (regular hours, recent months) and the US economic calendar.

Lessons carried over from the nq-strategy project (see its CLAUDE.md):
  * a single huge bulk export can be silently truncated -> download one YEAR at
    a time and check that each file covers the period we asked for;
  * the bulk export has a daily quota (HTTP 429 "too many export requests")
    -> fall back to the page-by-page `candles` endpoint, which always finishes;
  * cache every finished year on disk, so an interrupted run simply resumes.

All timestamps are stored in UTC and converted to New York time when loaded.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

COLS = ["timestamp", "open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Connecting
# ---------------------------------------------------------------------------
def load_key() -> str:
    """The key saved by `lse auth` lives in ~/.lse/config.json (the library
    itself does not read that file, so we do it here)."""
    env = os.environ.get("LSE_API_KEY")
    if env:
        return env
    cfg = Path.home() / ".lse" / "config.json"
    if cfg.exists():
        key = json.loads(cfg.read_text()).get("api_key", "")
        if key:
            return key
    raise SystemExit("No LSE API key found. Run:  lse auth YOUR_KEY_HERE")


def connect():
    from lse import LSE
    return LSE(api_key=load_key())


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------
def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Same columns and types whatever endpoint the data came from."""
    if "timestamp" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "timestamp"})
    elif "ts" in df.columns:
        df = df.drop(columns="ts")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[COLS].dropna(subset=["timestamp", "close"])
    # keep="last": when a re-download overlaps what we had, the newer copy wins
    return df.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


def _paged(client, symbol: str, dataset: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Page-by-page download: 3-day windows (< 5,000 one-minute bars each),
    ~90 calls per minute, which stays under the vendor's limit."""
    frames, cur, calls = [], start, 0
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=3), end)
        for attempt in range(5):
            try:
                rows = client.candles(symbol, "1m", start=cur.strftime("%Y-%m-%d"),
                                      end=nxt.strftime("%Y-%m-%d"), limit=5000, dataset=dataset)
                if rows:
                    if len(rows) >= 5000:
                        print(f"      !! {cur.date()}: hit the 5,000-row cap, window may be cut")
                    frames.append(pd.DataFrame(rows))
                break
            except Exception as exc:
                if attempt == 4:
                    print(f"      !! giving up on {cur.date()}: {exc}")
                time.sleep(2 ** attempt)
        calls += 1
        if calls % 40 == 0:
            print(f"      ... {cur.date()} ({sum(len(f) for f in frames):,} rows)", flush=True)
        cur = nxt
        time.sleep(0.65)
    return normalise(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame(columns=COLS)


def _fetch(client, symbol: str, dataset: str, start: pd.Timestamp, end: pd.Timestamp,
           state: dict) -> pd.DataFrame:
    """Try the fast bulk export first; once its daily quota is used up, switch
    to the page-by-page endpoint for the rest of the run."""
    if not state.get("bulk_blocked"):
        try:
            tmp = config.DATA_LSE.parent / "_tmp"
            tmp.mkdir(exist_ok=True)
            path = client.history(symbol, dataset=dataset, timeframe="1m",
                                  start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                                  dest=str(tmp), dataframe=False, timeout=3600.0)
            df = normalise(pd.read_parquet(path))
            os.remove(path)
            return df
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "too many" in msg.lower():
                print("    bulk-export quota used up for today -> switching to page-by-page download")
                state["bulk_blocked"] = True
            else:
                print(f"    bulk export failed ({msg[:120]}) -> trying page-by-page")
    print(f"    page-by-page {start.date()} -> {end.date()}", flush=True)
    return _paged(client, symbol, dataset, start, end)


def resolve_dataset(catalog: list[dict], symbol: str) -> str | None:
    """Which asset class holds `symbol` (ignoring options, which also list ETFs)."""
    found = sorted({r["dataset"] for r in catalog if r.get("symbol") == symbol})
    picks = [d for d in found if d != "options"] or found
    return picks[0] if picks else None


def update_symbol(client, name: str, symbol: str, dataset: str,
                  start: str = config.LSE_START, state: dict | None = None) -> None:
    """Bring data/raw/lse/<symbol>_<year>.parquet up to date, one year at a time.

    * finished years already on disk are skipped;
    * the current year (or an unfinished one) is topped up from its last bar;
    * every year is checked: a file that stops well before the year ends is flagged.
    """
    state = state if state is not None else {}
    today = pd.Timestamp.now(tz="UTC").normalize()
    first = pd.Timestamp(start, tz="UTC")
    for year in range(first.year, today.year + 1):
        path = config.DATA_LSE / f"{symbol.replace('/', '_')}_{year}.parquet"
        y_start = max(pd.Timestamp(f"{year}-01-01", tz="UTC"), first)
        y_end = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), today + pd.Timedelta(days=1))
        old = None
        if path.exists():
            old = pd.read_parquet(path)
            last = pd.to_datetime(old["timestamp"], utc=True).max()
            finished = year < today.year and last >= pd.Timestamp(f"{year}-12-24", tz="UTC")
            if finished:
                print(f"  {name} {year}: on disk ({len(old):,} rows) - skipped")
                continue
            y_start = last.normalize()
        print(f"  {name} {year}: downloading {y_start.date()} -> {y_end.date()}", flush=True)
        new = _fetch(client, symbol, dataset, y_start, y_end, state)
        parts = [df for df in (old, new) if df is not None and len(df)]
        if not parts:
            print(f"  !! {name} {year}: no data returned")
            continue
        merged = normalise(pd.concat(parts, ignore_index=True))
        merged.to_parquet(path, index=False)
        last = merged["timestamp"].max()
        flag = ""
        if year < today.year and last < pd.Timestamp(f"{year}-12-24", tz="UTC"):
            flag = "   !! ends early - check this year"
        print(f"  {name} {year}: saved {len(merged):,} rows, "
              f"{merged['timestamp'].min()} -> {last}{flag}", flush=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_lse(names: list[str], field: str = "close", how: str = "inner") -> pd.DataFrame:
    """1-minute `field` for each instrument (our short names, e.g. ["NQ", "ES"]),
    one column each, indexed by bar START time in New York time."""
    cols = {}
    for name in names:
        symbol = config.LSE_SYMBOLS[name]
        files = sorted(config.DATA_LSE.glob(f"{symbol.replace('/', '_')}_*.parquet"))
        if not files:
            raise FileNotFoundError(f"No LSE files for {name} ({symbol}) in {config.DATA_LSE} "
                                    f"- run 01_download_data.py first")
        df = normalise(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))
        s = df.set_index("timestamp")[field]
        s.index = s.index.tz_convert(config.TZ)
        s.index.name = "datetime"
        cols[name] = s
    return pd.concat(cols, axis=1, join=how).sort_index()


# ---------------------------------------------------------------------------
# Economic calendar (US macro releases: CPI, payrolls, FOMC, ...)
# ---------------------------------------------------------------------------
def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """A Parquet column must hold one type. API fields sometimes mix numbers and
    text (e.g. actual = 3.1 in one row and "3.1%" in another) or hold dicts/lists:
    store those columns as text; we parse them when we use them."""
    df = df.copy()

    def as_text(v):
        if isinstance(v, (dict, list, tuple)):
            return json.dumps(v, default=str)
        return None if pd.isna(v) else str(v)

    for c in df.columns:
        if df[c].dtype == object and not df[c].dropna().map(lambda v: isinstance(v, str)).all():
            df[c] = df[c].map(as_text)
    return df


def _calendar_window(client, start: pd.Timestamp, end: pd.Timestamp, region: str) -> list[dict]:
    """One request; if it comes back full (5,000 rows) the window is split in two,
    so nothing is silently cut off."""
    for attempt in range(5):
        try:
            rows = client.economic_calendar(region=region, start=start.strftime("%Y-%m-%d"),
                                            end=end.strftime("%Y-%m-%d"), limit=5000)
            break
        except Exception as exc:
            if attempt == 4:
                raise
            print(f"    calendar retry ({exc})")
            time.sleep(2 ** attempt)
    time.sleep(0.65)
    if len(rows) >= 5000 and (end - start).days > 1:
        mid = (start + (end - start) / 2).normalize()
        return _calendar_window(client, start, mid, region) + _calendar_window(client, mid, end, region)
    if len(rows) >= 5000:
        print(f"    !! calendar {start.date()}: 5,000 events in one day - list may be cut")
    return rows


def download_calendar(client, start: str = config.LSE_START, region: str = "US") -> pd.DataFrame:
    """Download the macro calendar quarter by quarter and save it as one file."""
    first = pd.Timestamp(start).normalize()
    stop = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    edges = pd.date_range(first, stop, freq="QS")
    bounds = [first] + [e for e in edges if e > first] + [stop]
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        part = _calendar_window(client, a, b, region)
        rows.extend(part)
        print(f"  calendar {a.date()} -> {b.date()}: {len(part):,} events", flush=True)
    if not rows:
        print("  !! the calendar came back empty - tell Claude")
        return pd.DataFrame()
    cal = _parquet_safe(pd.DataFrame(rows)).drop_duplicates()
    cal.to_parquet(config.ECON_CALENDAR, index=False)
    print(f"  saved {len(cal):,} events to {config.ECON_CALENDAR.name}")
    print("  columns:", list(cal.columns))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(cal.head(5).to_string())
    return cal


# ---------------------------------------------------------------------------
# 1-second bars, regular hours only, one file per instrument-day
# ---------------------------------------------------------------------------
# The vault documents start/end with dates ("2026-07-01"). For intraday windows
# we try the usual timestamp spellings once and keep the first one it honours.
_TIME_FORMATS = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]
_SECONDS_WINDOW = pd.Timedelta(minutes=80)       # 4,800 one-second bars < 5,000-row cap


def _seconds_path(symbol: str, day: pd.Timestamp) -> Path:
    return config.DATA_LSE_1S / f"{symbol.replace('/', '_')}_{day.date()}.parquet"


def _rth_windows_utc(day: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """09:30-16:00 New York on this day, cut into 80-minute pieces, in UTC."""
    open_ny = pd.Timestamp(f"{day.date()} 09:30", tz=config.TZ)
    close_ny = pd.Timestamp(f"{day.date()} 16:00", tz=config.TZ)
    edges = list(pd.date_range(open_ny, close_ny, freq=_SECONDS_WINDOW))
    if edges[-1] < close_ny:
        edges.append(close_ny)
    return [(a.tz_convert("UTC"), b.tz_convert("UTC")) for a, b in zip(edges[:-1], edges[1:])]


def _candles_1s(client, symbol: str, dataset: str, a: pd.Timestamp, b: pd.Timestamp,
                fmt: str) -> tuple[pd.DataFrame, int]:
    """1-second bars with a <= time < b. Network errors and rate limits are retried;
    a rejected request (bad filter) is raised straight away. Returns the rows inside
    the window and how many rows the server sent in total."""
    for attempt in range(5):
        try:
            rows = client.candles(symbol, "1s", start=a.strftime(fmt), end=b.strftime(fmt),
                                  limit=5000, dataset=dataset)
            break
        except Exception as exc:
            if getattr(exc, "status", 0) in (400, 401, 403, 404, 422) or attempt == 4:
                raise
            time.sleep(2 ** attempt)
    time.sleep(0.65)                                   # stay under ~100 calls per minute
    if not rows:
        return pd.DataFrame(columns=COLS), 0
    df = normalise(pd.DataFrame(rows))
    return df[(df["timestamp"] >= a) & (df["timestamp"] < b)].reset_index(drop=True), len(df)


def _find_time_format(client, symbol: str, dataset: str, days: list, state: dict) -> str | None:
    """Ask for 10 minutes of a recent trading day with each timestamp spelling and
    keep the first one the server honours. Tries up to 5 recent days (the newest
    one may not be published yet). None = no usable 1-second data."""
    known = state.get("time_format")
    formats = ([known] if known else []) + [f for f in _TIME_FORMATS if f != known]
    outside, errors = False, {}
    for day in list(days)[::-1][:5]:
        a = pd.Timestamp(f"{day.date()} 10:00", tz=config.TZ).tz_convert("UTC")
        b = a + pd.Timedelta(minutes=10)
        for fmt in formats:
            try:
                inside, n = _candles_1s(client, symbol, dataset, a, b, fmt)
            except Exception as exc:
                errors[fmt] = str(exc)[:80]
                continue
            if len(inside):
                state["time_format"] = fmt
                return fmt
            outside |= n > 0
        if outside:
            state["time_filter_ignored"] = True
            return None
    for fmt, err in errors.items():
        print(f"    (time format {fmt!r} not accepted: {err})")
    return None


def _trading_days(symbol: str, start: str) -> pd.Series:
    """Last regular-hours 1-minute bar (UTC) of every finished trading day since
    `start`, read from the 1-minute files on disk. Holidays are simply absent."""
    from .data import rth_mask
    slug = symbol.replace("/", "_")
    files = [f for f in sorted(config.DATA_LSE.glob(f"{slug}_*.parquet"))
             if int(f.stem.rsplit("_", 1)[1]) >= pd.Timestamp(start).year]
    if not files:
        return pd.Series(dtype="datetime64[ns, UTC]")
    ts = pd.concat([pd.read_parquet(f, columns=["timestamp"]) for f in files])["timestamp"]
    ny = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).tz_convert(config.TZ)
    ny = ny[rth_mask(ny)]
    last = pd.Series(ny, index=ny.tz_localize(None).normalize()).groupby(level=0).max()
    today = pd.Timestamp.now(tz=config.TZ).normalize().tz_localize(None)
    last = last[(last.index >= pd.Timestamp(start)) & (last.index < today)]
    return last.dt.tz_convert("UTC")


def update_seconds(client, name: str, symbol: str, dataset: str,
                   start: str = config.SECONDS_START, state: dict | None = None) -> None:
    """Regular-hours 1-second bars, one file per trading day, from `start` to yesterday.

    * trading days come from the 1-minute data (so run the 1-minute update first);
    * a day is saved only if it is complete (its last 1-second bar reaches the
      last 1-minute bar of that day); otherwise it is tried again next run;
    * finished days are cached, so re-running only fetches what is missing.
    """
    state = state if state is not None else {}
    last_bar = _trading_days(symbol, start)
    if last_bar.empty:
        print(f"  {name}: no 1-minute data since {start} on disk - run the 1-minute update first")
        return
    todo = [d for d in last_bar.index if not _seconds_path(symbol, d).exists()]
    if not todo:
        print(f"  {name}: all {len(last_bar)} days already on disk")
        return
    fmt = _find_time_format(client, symbol, dataset, todo, state)
    if fmt is None:
        if state.get("time_filter_ignored"):
            print("  !! LSE ignored the time of day in the request, so the 1-second download"
                  "\n     stopped. Tell Claude - a different download method is needed.")
        else:
            print(f"  !! {name}: LSE sent no 1-second bars for the last {min(5, len(todo))} trading days"
                  f" - skipped. Tell Claude.")
        return
    print(f"  {name}: {len(last_bar) - len(todo)} days on disk, downloading {len(todo)} "
          f"({todo[0].date()} -> {todo[-1].date()})", flush=True)
    new, missing = 0, []
    for i, day in enumerate(todo, 1):
        try:
            parts = [_candles_1s(client, symbol, dataset, a, b, fmt)[0] for a, b in _rth_windows_utc(day)]
        except Exception as exc:
            print(f"    !! {day.date()}: request failed ({str(exc)[:80]}) - retried next run")
            missing.append(day)
            continue
        parts = [p for p in parts if len(p)]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
        if df.empty or df["timestamp"].max() < last_bar[day] - pd.Timedelta(minutes=5):
            missing.append(day)          # not published (fully) yet: try again next run
            continue
        df.to_parquet(_seconds_path(symbol, day), index=False)
        new += 1
        if i % 10 == 0 or i == len(todo):
            print(f"    {name}: {i}/{len(todo)} days ({day.date()}: {len(df):,} bars)", flush=True)
    msg = f"  {name}: {len(last_bar) - len(missing)}/{len(last_bar)} days on disk ({new} new)"
    if missing:
        msg += (f"\n  !! {len(missing)} day(s) incomplete or missing, retried next run: "
                f"{', '.join(str(d.date()) for d in missing[:5])}{' ...' if len(missing) > 5 else ''}")
    print(msg, flush=True)


def load_seconds(names: list[str], field: str = "close") -> pd.DataFrame:
    """1-second `field` for each instrument, outer-joined (a second with no trade in
    one instrument is NaN there), index in New York time."""
    cols = {}
    for name in names:
        symbol = config.LSE_SYMBOLS[name]
        files = sorted(config.DATA_LSE_1S.glob(f"{symbol.replace('/', '_')}_*.parquet"))
        parts = [p for p in (pd.read_parquet(f) for f in files) if len(p)]
        if not parts:
            raise FileNotFoundError(f"No 1-second files for {name} - run 01_download_data.py first")
        df = normalise(pd.concat(parts, ignore_index=True))
        s = df.set_index("timestamp")[field]
        s.index = s.index.tz_convert(config.TZ)
        s.index.name = "datetime"
        cols[name] = s
    return pd.concat(cols, axis=1, join="outer").sort_index()
