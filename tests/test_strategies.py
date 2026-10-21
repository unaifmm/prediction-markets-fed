"""The trading tests of steps 5 and 6: fees, which trades count as the ask or the bid, locked-in profit."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(script: str):
    spec = importlib.util.spec_from_file_location(script.replace(".py", "_mod"), ROOT / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lag():
    return load("05_lag_trade.py")


@pytest.fixture(scope="module")
def venues():
    return load("06_venues.py")


def test_kalshi_fee_is_175_cents_per_contract_at_50(lag):
    assert abs(lag.kalshi_fee(0.5) - 0.0175) < 1e-12
    assert lag.kalshi_fee(0.0) == 0 and lag.kalshi_fee(1.0) == 0


def test_trades_are_split_into_ask_and_bid_prints(lag):
    km = pd.DataFrame({"event": ["K"], "bp": [0.0], "ticker": ["K-H0"]})
    kt = pd.DataFrame({"ticker": ["K-H0", "K-H0"], "ts": [1.0, 2.0], "yes_price": [0.52, 0.50],
                       "count": [10, 20], "taker_side": ["yes", "no"]})
    ask, bid, _ = lag.prints("kalshi", pd.Series({"kalshi_event": "K"}), 0.0, km, None, kt, None)
    assert list(ask["p"]) == [0.52] and list(bid["p"]) == [0.50]   # taker bought YES = ask, sold YES = bid

    pm = pd.DataFrame({"event": ["P"], "bp": [0.0], "condition_id": ["c"], "fee_rate": [0.05]})
    pt = pd.DataFrame({"conditionId": ["c"] * 3, "timestamp": [1, 2, 3], "outcome": ["Yes", "No", "No"],
                       "side": ["BUY", "SELL", "BUY"], "price": [0.52, 0.48, 0.51], "size": [1.0, 1.0, 1.0]})
    ask, bid, fee = lag.prints("polymarket", pd.Series({"poly_event": "P"}), 0.0, None, pm, None, pt)
    assert list(ask["p"]) == pytest.approx([0.52, 0.52])            # buying YES, or selling NO at 0.48
    assert list(bid["p"]) == pytest.approx([0.49])                  # buying NO at 0.51 = selling YES at 0.49
    assert fee(0.5) == pytest.approx(0.05 * 0.25)


def test_locked_in_profit_of_a_cross_venue_pair(venues):
    d = pd.DataFrame({"bid": [0.39, 0.44], "ask": [0.40, 0.45], "poly": [0.45, 0.45], "poly_fee_rate": [0.0, 0.0],
                      "days_to_decision": [30.0, 30.0], "ts": [1.7e9, 1.7e9]})
    out = venues.locked_in(d, half_spread=0.005)
    # row 0: YES on Kalshi at 0.40 + NO on Polymarket at 0.555 -> pays $1; Kalshi fee 0.07*0.4*0.6
    assert out["profit"].iloc[0] == pytest.approx(1 - 0.40 - 0.555 - 0.07 * 0.4 * 0.6)
    assert out["profit"].iloc[1] < 0                                # same price on both venues: costs only
