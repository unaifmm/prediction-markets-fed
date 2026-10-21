"""
One row per FOMC meeting, with each venue's event mapped to it.

Kalshi gives the exact decision time of each meeting (the event's strike date).
Polymarket gives an end date that can be the day after, so each Polymarket
event is matched to the Kalshi meeting within 3 days of its end date. When a
meeting has more than one Polymarket event, the one with the most volume is
the main one; the others are kept but flagged.
"""
from __future__ import annotations

import pandas as pd


def build(kalshi_markets: pd.DataFrame, poly_markets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (meetings, poly_markets with a 'decision_time' and 'main_event' column)."""
    k = (kalshi_markets.groupby("event")
         .agg(decision_time=("decision_time", "first"), kalshi_markets=("ticker", "count"),
              kalshi_volume=("volume", "sum"))
         .reset_index().rename(columns={"event": "kalshi_event"}))
    k["decision_time"] = pd.to_datetime(k["decision_time"], utc=True)
    k = k.sort_values("decision_time").reset_index(drop=True)

    pm = poly_markets.copy()
    if len(pm):
        ev = (pm.groupby(["event", "event_id"])
              .agg(event_end=("event_end", "first"), poly_volume=("volume", "sum"), poly_markets=("market_id", "count"))
              .reset_index())
        ev["event_end"] = pd.to_datetime(ev["event_end"], utc=True)

        def nearest(t):
            if pd.isna(t) or not len(k):
                return pd.NaT
            gap = (k["decision_time"] - t).abs()
            return k.loc[gap.idxmin(), "decision_time"] if gap.min() <= pd.Timedelta(days=3) else pd.NaT

        ev["decision_time"] = ev["event_end"].map(nearest)
        ev = ev.dropna(subset=["decision_time"])
        ev["main_event"] = ev["poly_volume"] == ev.groupby("decision_time")["poly_volume"].transform("max")
        pm = pm.merge(ev[["event_id", "decision_time", "main_event"]], on="event_id", how="inner")
        main = ev[ev["main_event"]].rename(columns={"event": "poly_event", "event_id": "poly_event_id"})
        meetings = k.merge(main[["decision_time", "poly_event", "poly_event_id", "poly_markets", "poly_volume"]],
                           on="decision_time", how="left")
    else:
        meetings = k.assign(poly_event=None, poly_event_id=None, poly_markets=0, poly_volume=0.0)
    meetings.insert(0, "meeting", meetings["decision_time"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d"))
    return meetings, pm
