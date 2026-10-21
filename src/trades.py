"""
Prediction-market trades around a release: exact times and prices.

Used for the second-by-second analyses (steps 4 and 6). Each outcome's
probability is its latest TRADE price; a Polymarket trade in the NO token at
price q means YES = 1 - q. Before the first trade in the window, an outcome
keeps its minute price from 5 minutes before the release.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, probs

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def unix(t) -> np.ndarray:
    """Datetimes -> Unix seconds (works whatever time unit pandas stores)."""
    return ((pd.to_datetime(t, utc=True) - EPOCH) / pd.Timedelta(seconds=1)).to_numpy()


def load(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(Kalshi trades with a 'ts' column in Unix seconds, Polymarket trades) of one release window."""
    kt = pd.read_parquet(config.KALSHI_DIR / "trades" / f"{tag}.parquet")
    pt = pd.read_parquet(config.POLY_DIR / "trades" / f"{tag}.parquet")
    if len(kt):
        kt["ts"] = unix(kt["time"])
    return kt, pt


def outcome_trades(venue: str, m: pd.Series, bp: float, km, pm, kt, pt) -> pd.DataFrame:
    """Trades in one outcome of meeting m: columns ts (Unix seconds) and p (YES price)."""
    if venue == "kalshi":
        tickers = km[(km["event"] == m["kalshi_event"]) & (km["bp"] == bp)]["ticker"]
        x = kt[kt["ticker"].isin(tickers)] if len(kt) else kt
        return pd.DataFrame({"ts": x["ts"], "p": x["yes_price"]}) if len(x) else pd.DataFrame(columns=["ts", "p"])
    cond = pm[(pm["event"] == m["poly_event"]) & (pm["bp"] == bp)]["condition_id"]
    x = pt[pt["conditionId"].isin(cond)] if len(pt) else pt
    if not len(x):
        return pd.DataFrame(columns=["ts", "p"])
    return pd.DataFrame({"ts": x["timestamp"].astype(float),
                         "p": np.where(x["outcome"] == "Yes", x["price"], 1 - x["price"])})


def trade_path(venue: str, tag: str, m: pd.Series, km, pm, kt, pt, t0: pd.Timestamp,
               grid: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """(expected rate change of meeting m on the grid from trade prices,
    Unix times of all trades in that meeting's outcomes)."""
    start = int(t0.timestamp()) - 300
    s = probs.minute_series(venue, tag, m, km, pm, start_ts=int((t0 - config.WINDOW_BEFORE).timestamp()))
    if not s:
        return None, np.array([])
    p0 = probs.on_grid(s, np.array([start]), max_age=3 * 86400).iloc[0]
    series, times = {}, []
    for bp, first in p0.items():
        x = outcome_trades(venue, m, bp, km, pm, kt, pt)
        pts = ([(start - 1, first)] if np.isfinite(first) else []) + list(zip(x["ts"], x["p"]))
        times += list(x["ts"])
        if pts:
            a = np.array(pts, dtype=float)
            a = a[np.argsort(a[:, 0], kind="stable")]
            series[bp] = pd.Series(a[:, 1], index=a[:, 0].astype("int64")).groupby(level=0).last()
    e = probs.summarise(probs.on_grid(series, grid, max_age=10 ** 9))["exp_bp"].to_numpy() if series else None
    ok = e is not None and np.isfinite(e).all()
    return (e if ok else None), np.sort(np.array(times, dtype=float))
