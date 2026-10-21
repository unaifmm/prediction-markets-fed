"""The downloaders, against a fake API: request limits, fallbacks, paging, privacy."""
import pandas as pd

from src import kalshi, polymarket


class FakeApi:
    """Answers get(path, params) from a function and records every call."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def get(self, path, params=None, **_):
        self.calls.append((path, dict(params or {})))
        return self.answer(path, params or {})


def candle(ts):
    return {"end_period_ts": ts, "volume_fp": "1.00",
            "price": {"close_dollars": "0.5000"}, "yes_bid": {"close_dollars": "0.4900"},
            "yes_ask": {"close_dollars": "0.5100"}}


def test_kalshi_candles_stay_under_the_request_cap():
    def answer(path, p):
        n = (p["end_ts"] - p["start_ts"]) // 60
        assert n <= kalshi.MAX_CANDLES                      # never ask for more than the cap
        return 200, {"candlesticks": [candle(p["start_ts"] + 60 * (i + 1)) for i in range(n)]}
    api = FakeApi(answer)
    df = kalshi.candles(api, "KXFEDDECISION", "X", 0, 10 * 86400, 1)
    assert len(api.calls) == 3                              # 14,400 minutes in 3 requests
    assert len(df) == 14400 and df["ts"].is_unique


def test_kalshi_candles_fall_back_to_the_historical_endpoint():
    def answer(path, p):
        if path.startswith("/series/"):
            return 404, {"error": "not found"}
        return 200, {"candlesticks": [candle(p["start_ts"] + 60)]}
    api = FakeApi(answer)
    df = kalshi.candles(api, "KXFEDDECISION", "X", 0, 3600, 1)
    assert len(df) == 1 and api.calls[-1][0].startswith("/historical/")


def test_kalshi_ticker_with_a_special_character_is_escaped_in_the_url():
    api = FakeApi(lambda path, p: (200, {"candlesticks": []}))
    kalshi.candles(api, "FEDDECISION", "FEDDECISION-23JUN-H>25", 0, 3600, 60)
    assert ">" not in api.calls[0][0] and "H%3E25" in api.calls[0][0]


def test_kalshi_trades_follow_the_cursor():
    pages = {None: ({"trades": [{"created_time": "2026-01-01T13:30:01Z", "ticker": "X", "yes_price_dollars": "0.40",
                                 "count_fp": "1", "taker_side": "yes", "trade_id": "a"}], "cursor": "next"}),
             "next": ({"trades": [{"created_time": "2026-01-01T13:30:02Z", "ticker": "X", "yes_price_dollars": "0.41",
                                   "count_fp": "1", "taker_side": "no", "trade_id": "b"}], "cursor": ""})}
    api = FakeApi(lambda path, p: (200, pages[p.get("cursor")]))
    df = kalshi.trades(api, "X", 0, 10 ** 10)
    assert list(df["trade_id"]) == ["a", "b"]


def test_polymarket_busy_window_is_split_and_wallets_are_dropped():
    def answer(path, p):
        span = p["end"] - p["start"]
        n = 10000 if span > 100 else 3                      # a full page until the window is small
        rows = [{"timestamp": p["start"] + i % max(span, 1), "conditionId": "c", "asset": "a", "outcome": "Yes",
                 "outcomeIndex": 0, "side": "BUY", "price": 0.5, "size": 1.0, "transactionHash": f"{p['start']}-{p['offset']}-{i}",
                 "proxyWallet": "0xabc", "name": "someone", "pseudonym": "x"} for i in range(n)]
        return 200, rows
    api = FakeApi(answer)
    df = polymarket.event_trades(api, "1", 0, 400)
    assert len(api.calls) > 2                               # the busy window was split
    assert not {"proxyWallet", "name", "pseudonym", "transactionHash"} & set(df.columns)


def test_polymarket_price_history_is_fetched_in_chunks():
    def answer(path, p):
        return 200, {"history": [{"t": p["startTs"], "p": 0.3}, {"t": p["endTs"], "p": 0.4}]}
    api = FakeApi(answer)
    df = polymarket.price_history(api, "tok", 0, 21 * 86400, fidelity=60, chunk_days=7)
    assert len(api.calls) == 3 and df["ts"].is_monotonic_increasing and df["ts"].is_unique


def test_polymarket_market_row_reads_the_yes_token():
    m = {"id": 5, "clobTokenIds": '["111", "222"]', "groupItemTitle": "No change", "volumeNum": 10}
    row = polymarket.market_row(m, {"slug": "fed-decision-in-may", "id": 9, "endDate": "2026-05-01T00:00:00Z"})
    assert row["yes_token"] == "111" and row["outcome"] == "No change" and row["event_id"] == "9"
