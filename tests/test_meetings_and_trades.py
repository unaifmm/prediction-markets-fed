"""Meetings mapping, price lookups and trade prices."""
import numpy as np
import pandas as pd

from src import meetings
from src.probs import next_meeting, on_grid
from src.trades import outcome_trades


def test_polymarket_events_are_matched_to_the_nearest_kalshi_meeting():
    km = pd.DataFrame({"event": ["K-SEP", "K-SEP", "K-OCT"], "ticker": ["a", "b", "c"], "volume": [1, 1, 1],
                       "decision_time": ["2026-09-16T18:00:00Z"] * 2 + ["2026-10-28T18:00:00Z"]})
    pm = pd.DataFrame({"event": ["sep-big", "sep-small", "oct", "far"], "event_id": ["1", "2", "3", "4"],
                       "market_id": ["m1", "m2", "m3", "m4"], "volume": [100.0, 5.0, 50.0, 1.0],
                       "event_end": ["2026-09-16T00:00:00Z", "2026-09-16T00:00:00Z", "2026-10-29T00:00:00Z",
                                     "2026-12-30T00:00:00Z"]})
    mt, pm2 = meetings.build(km, pm)
    assert list(mt["poly_event"]) == ["sep-big", "oct"]        # the bigger of two September events is the main one
    assert "far" not in set(pm2["event"])                       # no Kalshi meeting within 3 days: dropped
    assert pm2.set_index("event").loc["sep-small", "main_event"] == False  # noqa: E712


def test_on_grid_takes_the_latest_price_and_respects_its_age():
    s = {0.0: pd.Series([0.2, 0.3], index=[100, 200])}
    g = on_grid(s, np.array([50, 150, 200, 500]), max_age=100)
    assert np.isnan(g.iloc[0, 0])                               # nothing yet
    assert g.iloc[1, 0] == 0.2 and g.iloc[2, 0] == 0.3          # latest at or before
    assert np.isnan(g.iloc[3, 0])                               # 300 s old: too stale


def test_next_meeting_is_strictly_after_the_release():
    mt = pd.DataFrame({"meeting": ["a", "b"],
                       "decision_time": pd.to_datetime(["2026-09-16 18:00", "2026-10-28 18:00"], utc=True)})
    assert next_meeting(mt, pd.Timestamp("2026-09-16 18:00", tz="UTC"))["meeting"] == "b"
    assert next_meeting(mt, pd.Timestamp("2026-09-11 12:30", tz="UTC"))["meeting"] == "a"


def test_a_polymarket_no_trade_is_read_as_one_minus_its_price():
    pm = pd.DataFrame({"event": ["e"], "bp": [0.0], "condition_id": ["c"]})
    pt = pd.DataFrame({"conditionId": ["c", "c"], "timestamp": [1, 2], "outcome": ["Yes", "No"], "price": [0.7, 0.25]})
    x = outcome_trades("polymarket", pd.Series({"poly_event": "e"}), 0.0, None, pm, None, pt)
    assert list(x["p"]) == [0.7, 0.75]
