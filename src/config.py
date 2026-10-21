"""
Central configuration for project 2: prediction markets vs the Fed.

Every setting lives here: change it once and every script picks it up.
"""
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Folders (relative to this file, so the project works on any computer)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
KALSHI_DIR = RAW / "kalshi"
POLY_DIR = RAW / "polymarket"
FRED_DIR = RAW / "fred"
DATA_LSE = RAW / "lse"                       # 1-minute bars, one parquet file per instrument-year
DATA_LSE_1S = RAW / "lse_1s"                 # 1-second bars around each release, one file per market and release
LSE_CATALOG = RAW / "lse_catalog.csv"
ECON_CALENDAR = RAW / "economic_calendar_us.parquet"
MEETINGS_FILE = RAW / "meetings.csv"         # one row per FOMC meeting, both venues mapped to it
COVERAGE_FILE = RAW / "coverage.csv"         # what the download actually got, window by window
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

for _folder in (KALSHI_DIR / "hourly", KALSHI_DIR / "minute", KALSHI_DIR / "trades",
                POLY_DIR / "hourly", POLY_DIR / "minute", POLY_DIR / "trades",
                FRED_DIR, DATA_LSE, DATA_LSE_1S, FIGURES, RESULTS):
    _folder.mkdir(parents=True, exist_ok=True)

TZ = "America/New_York"

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
START = "2023-05-01"            # Kalshi's first Fed-decision market is the May 2023 meeting
HORIZON_DAYS = 130              # around each release, track the meetings in the next ~130 days
                                # (the next two or three decisions)

# Macro releases studied, and the window of minute data kept around each one
EVENT_KINDS = ("CPI", "Payrolls", "FOMC")
WINDOW_BEFORE = pd.Timedelta(hours=3)
WINDOW_AFTER = pd.Timedelta(hours=6)

# ---------------------------------------------------------------------------
# Prediction markets (public, read-only APIs: no account or key needed)
# ---------------------------------------------------------------------------
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES = "KXFEDDECISION"     # "Fed decision in <month>?": cut >25, cut 25, hold, hike 25, hike >25
                                    # (older events are listed under it with the FEDDECISION- prefix)
POLY_GAMMA = "https://gamma-api.polymarket.com"   # markets and events
POLY_CLOB = "https://clob.polymarket.com"         # price history
POLY_DATA = "https://data-api.polymarket.com"     # trades
POLY_QUERIES = ("Fed decision", "Fed interest rates", "FOMC")
# Polymarket titles of the monthly decision events, e.g. "Fed decision in October?",
# "Fed Interest Rates: March 2024" (other Fed markets - dissents, combos, "how many cuts" - are left out)
POLY_TITLE_PATTERN = r"^fed (interest rates:? [a-z]+ \d{4}|decision in [a-z]+\??)$"

# ---------------------------------------------------------------------------
# Benchmark markets (London Strategic Edge, 1-minute bars) and daily rates (FRED)
# ---------------------------------------------------------------------------
LSE_START = START
LSE_SYMBOLS = {                 # our name -> LSE symbol
    "ES": "ES.F",               # E-mini S&P 500 futures: the stock market's reaction
    "USDJPY": "USD/JPY",        # the dollar: very sensitive to US rate expectations
    "TLT": "TLT",               # 20+ year Treasury ETF
}
SECONDS_START = "2026-07-01"    # (not used here; kept so src/lse.py works unchanged)
SECONDS_FROM = "2025-01-01"     # 1-second bars of ES and USD/JPY around CPI and payrolls from this date
SECONDS_BEFORE = pd.Timedelta(minutes=5)
SECONDS_AFTER = pd.Timedelta(minutes=10)    # 900 one-second bars per request (the vendor's cap is 5,000)

FRED_SERIES = {                 # free daily series from the St. Louis Fed
    "DFF": "Effective federal funds rate",
    "SOFR": "Secured overnight financing rate",
    "DTB3": "3-month Treasury bill",
    "DTB6": "6-month Treasury bill",
    "DGS1": "1-year Treasury yield",
    "DGS2": "2-year Treasury yield",
    "DGS10": "10-year Treasury yield",
}
