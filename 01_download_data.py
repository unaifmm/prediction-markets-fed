"""
STEP 1 - Download everything the study needs (read-only, public data + your LSE key).

Run on YOUR computer, from this folder:

    python 01_download_data.py                  # everything (first run: about 30-60 minutes)
    python 01_download_data.py --parts kalshi   # only some parts (see PARTS below)

Safe to stop and re-run at any time: finished pieces are saved in data/raw/ and
skipped next time; only markets that are still trading are refreshed.

What it downloads
  1. calendar    US macro calendar (London Strategic Edge) -> release dates of CPI,
                 payrolls and FOMC decisions since 2023
  2. kalshi      every "Fed decision in <month>?" market since May 2023: metadata and
                 hourly prices over each market's whole life
  3. polymarket  the same for Polymarket's monthly Fed decision events
  4. windows     around every CPI, payrolls and FOMC release: 1-minute prices and every
                 trade on both venues, from 3 hours before to 6 hours after
  5. fred        daily US rates (fed funds, SOFR, T-bills, 1-, 2- and 10-year yields)
  6. lse         1-minute S&P 500 futures, USD/JPY and TLT (the fast markets to compare with)
  7. seconds     1-second S&P 500 futures and USD/JPY around every CPI and payrolls release
                 since 2025 (5 minutes before to 10 minutes after)

Everything is written under data/raw/ (not uploaded to GitHub). At the end a
coverage table shows, release by release, how much data came back.
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from src import config, events, fred, kalshi, meetings, polymarket
from src.webapi import Api

sys.stdout.reconfigure(errors="replace")
PARTS = ["calendar", "kalshi", "polymarket", "windows", "fred", "lse", "seconds"]


def save(df: pd.DataFrame, path) -> None:
    df.to_parquet(path, index=False)


def val(x):
    """None for missing values (NaN, None, empty text), so `if val(x)` is safe."""
    return None if x is None or (isinstance(x, float) and pd.isna(x)) or x == "" else x


def kalshi_markets(api: Api, start, horizon) -> pd.DataFrame:
    evs = kalshi.list_events(api, config.KALSHI_SERIES)
    evs = [e for e in evs if e.get("strike_date")
           and start <= pd.Timestamp(e["strike_date"]) <= horizon]
    rows = []
    for e in sorted(evs, key=lambda e: e["strike_date"]):
        ms, where = kalshi.event_markets(api, e["event_ticker"])
        rows += [kalshi.market_row(m, e, where) | {"series": e.get("series_ticker") or config.KALSHI_SERIES}
                 for m in ms]
        print(f"  {e['event_ticker']:22s} {e['strike_date'][:10]}  {len(ms)} markets ({where})")
    df = pd.DataFrame(rows)
    df.to_csv(config.KALSHI_DIR / "markets.csv", index=False)
    return df


def poly_markets(gamma: Api, start, horizon) -> pd.DataFrame:
    evs = polymarket.search_events(gamma, config.POLY_QUERIES, config.POLY_TITLE_PATTERN)
    rows = []
    for e in evs:
        end = pd.Timestamp(e.get("endDate")) if e.get("endDate") else None
        if end is None or not (start - pd.Timedelta(days=3) <= end <= horizon):
            continue
        ms = polymarket.event_markets(gamma, e)
        rows += [polymarket.market_row(m, e) for m in ms]
        print(f"  {str(e.get('endDate'))[:10]}  {e.get('title')[:40]:40s} {len(ms)} markets")
    df = pd.DataFrame(rows)
    df.to_csv(config.POLY_DIR / "markets.csv", index=False)
    return df


def hourly_kalshi(api: Api, km: pd.DataFrame, now, refresh: bool = False) -> None:
    print(f"\n=== Kalshi hourly prices: {len(km)} markets ===")
    for _, m in km.iterrows():
        path = config.KALSHI_DIR / "hourly" / f"{kalshi.file_name(m['ticker'])}.parquet"
        settled = str(val(m.get("result")) or "") in ("yes", "no")
        if (path.exists() and settled and not refresh) or not val(m.get("open_time")):
            continue
        end = min(pd.Timestamp(m["close_time"]), now) if val(m.get("close_time")) else now
        df = kalshi.candles(api, m["series"], m["ticker"], m["open_time"], end, 60,
                            historical=m["served_by"] == "historical")
        save(df, path)
        print(f"  {m['ticker']:30s} {len(df):6,} hours")


def hourly_poly(clob: Api, pm: pd.DataFrame, now) -> None:
    pm = pm[pm["main_event"]]
    print(f"\n=== Polymarket hourly prices: {len(pm)} markets ===")
    for _, m in pm.iterrows():
        if not val(m.get("yes_token")):
            continue
        path = config.POLY_DIR / "hourly" / f"{m['market_id']}.parquet"
        if path.exists() and str(m.get("closed")).lower() == "true":
            continue
        start = pd.Timestamp(m["start_time"]) if val(m.get("start_time")) else m["decision_time"] - pd.Timedelta(days=120)
        end = pd.Timestamp(m["closed_time"]) + pd.Timedelta(hours=1) if val(m.get("closed_time")) \
            else m["decision_time"] + pd.Timedelta(days=1)
        end = min(end, now)
        df = polymarket.price_history(clob, m["yes_token"], start, end, fidelity=60, chunk_days=7)
        save(df, path)
        print(f"  {m['event'][:38]:38s} {str(m['outcome'])[:18]:18s} {len(df):6,} hours")


def release_windows(kal: Api, clob: Api, data: Api, km, pm, cut, now, refresh: bool = False) -> pd.DataFrame:
    rel = events.load_events({k: events.RELEASES[k] for k in config.EVENT_KINDS})
    rel["time_utc"] = rel["time"].dt.tz_convert("UTC")
    rel = rel[(rel["time_utc"] >= pd.Timestamp(config.START, tz="UTC"))
              & (rel["time_utc"] + config.WINDOW_AFTER <= now)].reset_index(drop=True)
    print(f"\n=== Release windows: {len(rel)} releases "
          f"({', '.join(f'{k} {n}' for k, n in rel['kind'].value_counts().items())}) ===")
    km = km.assign(decision_time=pd.to_datetime(km["decision_time"], utc=True))
    cover = []
    for i, r in rel.iterrows():
        t0 = r["time_utc"]
        a, b = t0 - config.WINDOW_BEFORE, t0 + config.WINDOW_AFTER
        lo, hi = t0 - pd.Timedelta(hours=1), t0 + pd.Timedelta(days=config.HORIZON_DAYS)
        tag = f"{r['kind']}_{t0.tz_convert(config.TZ):%Y-%m-%d}"
        files = {k: getattr(config, d) / k.split("_")[1] / f"{tag}.parquet"
                 for k, d in (("k_minute", "KALSHI_DIR"), ("k_trades", "KALSHI_DIR"),
                              ("p_minute", "POLY_DIR"), ("p_trades", "POLY_DIR"))}
        kk = km[(km["decision_time"] >= lo) & (km["decision_time"] <= hi)]
        pp = pm[(pm["decision_time"] >= lo) & (pm["decision_time"] <= hi) & pm["main_event"]] if len(pm) else pm
        t_start = time.time()

        if refresh or not files["k_minute"].exists():
            hist = kk["served_by"] == "historical"
            parts = [kalshi.candles(kal, m["series"], m["ticker"], a, b, 1, historical=h)
                     for (_, m), h in zip(kk.iterrows(), hist)]
            save(pd.concat([p for p in parts if len(p)] or [pd.DataFrame()], ignore_index=True), files["k_minute"])
        if not files["k_trades"].exists():
            old = b.timestamp() < cut["trades_created_ts"]
            parts = [kalshi.trades(kal, m["ticker"], a, b, historical=old) for _, m in kk.iterrows()]
            save(pd.concat([p for p in parts if len(p)] or [pd.DataFrame()], ignore_index=True), files["k_trades"])
        if not files["p_minute"].exists():
            parts = [polymarket.price_history(clob, m["yes_token"], a, b, fidelity=1, chunk_days=1)
                     for _, m in pp.iterrows() if val(m.get("yes_token"))]
            save(pd.concat([p for p in parts if len(p)] or [pd.DataFrame()], ignore_index=True), files["p_minute"])
        if not files["p_trades"].exists():
            parts = [polymarket.event_trades(data, eid, a, b).assign(event_id=eid)
                     for eid in sorted(set(pp["event_id"])) if eid]
            save(pd.concat([p for p in parts if len(p)] or [pd.DataFrame()], ignore_index=True), files["p_trades"])

        n = {k: len(pd.read_parquet(f)) for k, f in files.items()}
        cover.append({"release": tag, "kind": r["kind"], "time_utc": t0, "kalshi_markets": len(kk),
                      "poly_markets": len(pp), **n})
        print(f"  [{i + 1:3d}/{len(rel)}] {tag:22s} kalshi {len(kk):2d} mkts {n['k_minute']:6,} min "
              f"{n['k_trades']:6,} trades | polymarket {len(pp):2d} mkts {n['p_minute']:6,} min "
              f"{n['p_trades']:6,} trades  ({time.time() - t_start:.0f}s)", flush=True)
    cov = pd.DataFrame(cover)
    cov.to_csv(config.COVERAGE_FILE, index=False)
    return cov


def seconds_windows(client, now) -> None:
    """1-second bars of ES and USD/JPY from 5 minutes before to 10 minutes after each
    CPI and payrolls release since SECONDS_FROM. One file per market and release."""
    from src.lse import _TIME_FORMATS, _candles_1s, resolve_dataset
    rel = events.load_events({k: events.RELEASES[k] for k in ("CPI", "Payrolls")})
    rel = rel[(rel["time"] >= pd.Timestamp(config.SECONDS_FROM, tz=config.TZ))
              & (rel["time"] + config.SECONDS_AFTER <= now)].reset_index(drop=True)
    catalog = (pd.read_csv(config.LSE_CATALOG).to_dict("records") if config.LSE_CATALOG.exists()
               else client.datasets())
    print(f"\n=== 1-second bars around {len(rel)} CPI and payrolls releases since {config.SECONDS_FROM} ===")
    fmt = None
    for name in ("ES", "USDJPY"):
        symbol = config.LSE_SYMBOLS[name]
        ds = resolve_dataset(catalog, symbol)
        got = 0
        for _, r in rel.iterrows():
            tag = f"{r['kind']}_{r['time']:%Y-%m-%d}"
            path = config.DATA_LSE_1S / f"{name}_{tag}.parquet"
            if path.exists():
                got += len(pd.read_parquet(path)) > 0
                continue
            a = (r["time"] - config.SECONDS_BEFORE).tz_convert("UTC")
            b = (r["time"] + config.SECONDS_AFTER).tz_convert("UTC")
            df, answered = pd.DataFrame(), False
            for f in ([fmt] if fmt else []) + [x for x in _TIME_FORMATS if x != fmt]:
                try:
                    df, _ = _candles_1s(client, symbol, ds, a, b, f)
                    answered = True
                except Exception as exc:
                    print(f"    ({name} {tag}: time format {f!r} refused: {str(exc)[:60]})")
                    continue
                if len(df):
                    fmt = f
                    break
            if answered:                     # an error is retried next run; "no data" is kept
                save(df, path)
            got += bool(len(df))
            print(f"  {name:6s} {tag:22s} {len(df):4d} seconds")
        print(f"  {name}: {got} of {len(rel)} releases have 1-second data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=",".join(PARTS), help="comma-separated, from: " + ", ".join(PARTS))
    ap.add_argument("--refresh-kalshi", action="store_true",
                    help="download Kalshi prices (hourly and 1-minute) again even if they are on disk")
    args = ap.parse_args()
    parts = [p.strip() for p in args.parts.split(",")]
    now = pd.Timestamp.now(tz="UTC")
    start = pd.Timestamp(config.START, tz="UTC")
    horizon = now + pd.Timedelta(days=config.HORIZON_DAYS)

    client = None
    if "calendar" in parts or "lse" in parts or "seconds" in parts:
        from src.lse import connect
        client = connect()

    if "calendar" in parts:
        print("=== US macro calendar (London Strategic Edge) ===")
        from src.lse import download_calendar
        download_calendar(client, start="2023-01-01")

    kal = Api(config.KALSHI_BASE, pause=0.1)
    gamma, clob, data = Api(config.POLY_GAMMA, 0.1), Api(config.POLY_CLOB, 0.1), Api(config.POLY_DATA, 0.15)
    km = pm = None
    if any(p in parts for p in ("kalshi", "polymarket", "windows")):
        print("\n=== Kalshi: Fed decision markets ===")
        km = kalshi_markets(kal, start, horizon)
        print("\n=== Polymarket: Fed decision events ===")
        pm = poly_markets(gamma, start, horizon)
        mt, pm = meetings.build(km, pm)
        mt.to_csv(config.MEETINGS_FILE, index=False)
        print(f"\n{len(mt)} meetings: {mt['meeting'].iloc[0]} -> {mt['meeting'].iloc[-1]}; "
              f"{mt['poly_event'].notna().sum()} also on Polymarket")
    if "kalshi" in parts:
        hourly_kalshi(kal, km, now, refresh=args.refresh_kalshi)
    if "polymarket" in parts:
        hourly_poly(clob, pm, now)
    if "windows" in parts:
        cut = kalshi.cutoffs(kal)
        cov = release_windows(kal, clob, data, km, pm, cut, now, refresh=args.refresh_kalshi)
        empty = cov[(cov["k_minute"] == 0) | (cov["p_minute"] == 0)]
        print(f"\nWindows with no 1-minute prices on a venue: {len(empty)} of {len(cov)}"
              + (" (list in data/raw/coverage.csv)" if len(empty) else ""))
    if "fred" in parts:
        print("\n=== FRED daily rates ===")
        fred.download()
    if "lse" in parts:
        print("\n=== London Strategic Edge 1-minute benchmarks ===")
        from src.lse import resolve_dataset, update_symbol
        catalog = client.datasets()
        pd.DataFrame(catalog).to_csv(config.LSE_CATALOG, index=False)
        state = {}
        for name, symbol in config.LSE_SYMBOLS.items():
            ds = resolve_dataset(catalog, symbol)
            if ds is None:
                print(f"  !! {name}: '{symbol}' not in the LSE catalog - skipped")
                continue
            print(f"--- {name} ({symbol}, dataset '{ds}')")
            update_symbol(client, name, symbol, ds, start=config.LSE_START, state=state)

    if "seconds" in parts:
        seconds_windows(client, now)

    print(f"\nDone. API calls: Kalshi {kal.calls:,}, Polymarket {gamma.calls + clob.calls + data.calls:,}. "
          f"Tell Claude it has finished.")


if __name__ == "__main__":
    main()
