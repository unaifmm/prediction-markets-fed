"""Kalshi parsing: prices in the right units, safe file names."""
from src.kalshi import file_name, flatten_candles, flatten_trades

LIVE = {"end_period_ts": 100, "volume_fp": "5.00", "open_interest_fp": "10.00",
        "price": {"close_dollars": "0.4500", "open_dollars": "0.4000", "high_dollars": "0.5000",
                  "low_dollars": "0.4000", "mean_dollars": "0.4500"},
        "yes_bid": {"close_dollars": "0.4400"}, "yes_ask": {"close_dollars": "0.4600"}}
HISTORICAL = {"end_period_ts": 100, "volume": "5.00", "open_interest": "10.00",
              "price": {"close": "0.4500", "open": "0.4000", "high": "0.5000", "low": "0.4000", "mean": "0.4500"},
              "yes_bid": {"close": "0.4400"}, "yes_ask": {"close": "0.4600"}}
OLD_CENTS = {"end_period_ts": 100, "volume": 5, "price": {"close": 45}, "yes_bid": {"close": 44}, "yes_ask": {"close": 46}}


def test_candle_prices_are_dollars_in_every_format():
    for raw in (LIVE, HISTORICAL, OLD_CENTS):
        row = flatten_candles([raw]).iloc[0]
        assert abs(row["price_close"] - 0.45) < 1e-12
        assert abs(row["yes_bid_close"] - 0.44) < 1e-12
        assert abs(row["yes_ask_close"] - 0.46) < 1e-12


def test_trade_prices_are_dollars():
    t = flatten_trades([{"created_time": "2026-09-16T17:58:59Z", "ticker": "X", "yes_price_dollars": "0.0100",
                         "count_fp": "620.00", "taker_side": "yes", "trade_id": "a"},
                        {"created_time": "2023-06-14T17:00:00Z", "ticker": "Y", "yes_price": 3,
                         "count": 10, "taker_side": "no", "trade_id": "b"}])
    assert list(t["yes_price"]) == [0.03, 0.01]          # sorted by time: the 2023 trade (3 cents) first


def test_file_names_are_safe_on_windows():
    assert file_name("FEDDECISION-23JUN-H>25") == "FEDDECISION-23JUN-Hgt25"
    assert file_name("KXFEDDECISION-26SEP-H0") == "KXFEDDECISION-26SEP-H0"
