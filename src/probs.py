"""
From raw prices to probabilities and the market's expected rate change.

For one FOMC meeting each venue lists four to six outcomes (cut 50, cut 25,
hold, hike 25, ...). The price of each outcome's YES contract (0 to 1) reads as
its probability. From them we build ONE number per venue and moment:

    expected change (bp) = sum_k p_k * change_k / sum_k p_k

e.g. 85% hold and 15% hike 25 bp -> 0.85*0 + 0.15*25 = +3.75 bp. Dividing by
sum_k p_k removes the small "overround" (prices add up to slightly more than 1,
which is how market makers get paid).

Prices:
  * Kalshi   - the mid-point of the best bid and ask; if the quote is missing or
               wider than MAX_SPREAD, the last trade price instead.
  * Polymarket - the price history the API returns (one point per minute or hour).
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import config
from .kalshi import file_name

MAX_SPREAD = 0.10            # a quote wider than 10 cents says little about the probability


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------
def rate_change_bp(label) -> float:
    """Outcome label -> rate change in basis points, on either venue.
    'Cut >25bps' / '50+ bps decrease' -> -50; 'Hike 25bps' -> +25; 'No change' -> 0;
    'Hike >50bps' -> +75; 'Other' -> NaN (not used)."""
    s = str(label).lower().strip()
    if s in ("other", "", "nan"):
        return np.nan
    if re.search(r"no change|maintain|no cut|hike 0|^0 bps", s):
        return 0.0
    m = re.search(r"(\d+)", s)
    if not m:
        return np.nan
    n = float(m.group(1))
    if ">" in s:                       # "more than 25" -> the next step, 50
        n += 25
    if "cut" in s or "decrease" in s:
        return -n
    if "hike" in s or "increase" in s:
        return n
    return np.nan


def markets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(Kalshi markets, Polymarket markets of each meeting's main event, meetings),
    each with the rate change of the outcome ('bp') and times as UTC datetimes."""
    mt = pd.read_csv(config.MEETINGS_FILE)
    mt["decision_time"] = pd.to_datetime(mt["decision_time"], utc=True)
    km = pd.read_csv(config.KALSHI_DIR / "markets.csv")
    km["bp"] = km["outcome"].map(rate_change_bp)
    km["decision_time"] = pd.to_datetime(km["decision_time"], utc=True)
    pm = pd.read_csv(config.POLY_DIR / "markets.csv", dtype={"yes_token": str, "market_id": str, "event_id": str})
    pm["bp"] = pm["outcome"].map(rate_change_bp)
    pm = pm.merge(mt[["poly_event", "decision_time"]], left_on="event", right_on="poly_event", how="inner")
    pm["final_yes"] = pm["final_prices"].map(lambda s: float(json.loads(s)[0]) if isinstance(s, str) else np.nan)
    return km, pm, mt


# ---------------------------------------------------------------------------
# Price series per outcome
# ---------------------------------------------------------------------------
def kalshi_price(df: pd.DataFrame) -> pd.Series:
    """Probability from a Kalshi candle: quote mid-point, or last trade if the quote is poor."""
    bid, ask = df["yes_bid_close"], df["yes_ask_close"]
    good = (bid >= 0) & (ask > 0) & (ask >= bid) & (ask - bid <= MAX_SPREAD)
    return pd.Series(np.where(good, (bid + ask) / 2, df["price_close"]), index=df.index, dtype=float)


def _series(ts, p) -> pd.Series:
    s = pd.Series(np.asarray(p, dtype=float), index=np.asarray(ts, dtype="int64")).dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def hourly_series(venue: str, meeting: pd.Series, km: pd.DataFrame, pm: pd.DataFrame) -> dict[float, pd.Series]:
    """{rate change: hourly price series (index = Unix seconds)} for one meeting."""
    out = {}
    if venue == "kalshi":
        for _, m in km[km["event"] == meeting["kalshi_event"]].iterrows():
            f = config.KALSHI_DIR / "hourly" / f"{file_name(m['ticker'])}.parquet"
            d = pd.read_parquet(f) if f.exists() else pd.DataFrame()
            if len(d) and pd.notna(m["bp"]):
                out[m["bp"]] = _series(d["ts"], kalshi_price(d))
    else:
        for _, m in pm[pm["event"] == meeting["poly_event"]].iterrows():
            f = config.POLY_DIR / "hourly" / f"{m['market_id']}.parquet"
            d = pd.read_parquet(f) if f.exists() else pd.DataFrame()
            if len(d) and pd.notna(m["bp"]):
                out[m["bp"]] = _series(d["ts"], d["p"])
    return out


def on_grid(series: dict[float, pd.Series], grid: np.ndarray, max_age: int) -> pd.DataFrame:
    """Each outcome's latest price at or before every grid time (not older than max_age seconds)."""
    cols = {}
    for bp, s in series.items():
        if not len(s):
            continue
        pos = np.searchsorted(s.index.to_numpy(), grid, side="right") - 1
        val = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
        age = np.where(pos >= 0, grid - s.index.to_numpy()[np.clip(pos, 0, None)], np.inf)
        cols[bp] = np.where(age <= max_age, val, np.nan)
    return pd.DataFrame(cols, index=grid).sort_index(axis=1)


def summarise(probs: pd.DataFrame, min_share: float = 0.8, tail_ok: float = 0.97) -> pd.DataFrame:
    """Per grid time: sum of prices, expected change (bp) and probability of no change.
    A time counts if every outcome has a price, or if the outcomes that have one already
    add up to at least `tail_ok` (the missing ones are tails nobody quoted, e.g. a 50 bp
    hike nobody expects; they count as zero). The sum must also be at least `min_share`."""
    present = probs.notna()
    total = probs.sum(axis=1, min_count=1)
    complete = (present.all(axis=1) & (total >= min_share)) | (total >= tail_ok)
    exp = probs.fillna(0).mul(probs.columns.to_numpy(), axis=1).sum(axis=1) / total
    hold = probs.get(0.0, pd.Series(np.nan, index=probs.index)).fillna(0)
    return pd.DataFrame({"total": total.where(complete), "exp_bp": exp.where(complete),
                         "p_hold": hold.where(complete)})


# ---------------------------------------------------------------------------
# Minute data around a release
# ---------------------------------------------------------------------------
def minute_series(venue: str, tag: str, meeting: pd.Series, km: pd.DataFrame,
                  pm: pd.DataFrame, start_ts: int | None = None,
                  align_polymarket: bool = True) -> dict[float, pd.Series]:
    """{rate change: price series} of one meeting's outcomes in the window around release `tag`.
    Kalshi only reports minutes with activity, so each outcome's series starts from its
    last hourly price before the window (start_ts); Polymarket reports every minute.
    align_polymarket=False keeps Polymarket's own time stamps (a few seconds late)."""
    out = {}
    if venue == "kalshi":
        f = config.KALSHI_DIR / "minute" / f"{tag}.parquet"
        d = pd.read_parquet(f) if f.exists() else pd.DataFrame()
        for _, m in km[km["event"] == meeting["kalshi_event"]].iterrows():
            if pd.isna(m["bp"]):
                continue
            x = d[d["ticker"] == m["ticker"]] if len(d) else d
            parts = []
            fh = config.KALSHI_DIR / "hourly" / f"{file_name(m['ticker'])}.parquet"
            first = start_ts if start_ts is not None else (x["ts"].min() if len(x) else None)
            if fh.exists() and first is not None:
                h = pd.read_parquet(fh)
                h = h[h["ts"] <= first]
                if len(h):
                    parts.append(_series(h["ts"].iloc[-1:], kalshi_price(h.iloc[-1:])))
            if len(x):
                parts.append(_series(x["ts"], kalshi_price(x)))
            if parts:
                s = pd.concat(parts)
                out[m["bp"]] = s[~s.index.duplicated(keep="last")].sort_index()
    else:
        f = config.POLY_DIR / "minute" / f"{tag}.parquet"
        d = pd.read_parquet(f) if f.exists() else pd.DataFrame()
        if len(d):
            d["yes_token"] = d["yes_token"].astype(str)
        for _, m in pm[pm["event"] == meeting["poly_event"]].iterrows():
            x = d[d["yes_token"] == str(m["yes_token"])] if len(d) else d
            if len(x) and pd.notna(m["bp"]):
                # Polymarket stamps its 1-minute prices a few seconds after each minute starts
                # (mostly 2-11 s): read each as the price at the start of its minute
                stamp = x["ts"] // 60 * 60 if align_polymarket else x["ts"]
                out[m["bp"]] = _series(stamp, x["p"])
    return out


def next_meeting(mt: pd.DataFrame, t0: pd.Timestamp) -> pd.Series | None:
    """The first FOMC decision strictly after time t0."""
    later = mt[mt["decision_time"] > t0 + pd.Timedelta(minutes=1)]
    return later.iloc[0] if len(later) else None
