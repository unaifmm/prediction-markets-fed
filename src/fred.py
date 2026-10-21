"""
Daily interest rates from FRED (Federal Reserve Bank of St. Louis), free CSV
downloads with no key: the bond market's view to compare the prediction
markets with.
"""
from __future__ import annotations

import io

import pandas as pd

from . import config
from .webapi import Api


def download(series: dict = config.FRED_SERIES, start: str = config.START) -> pd.DataFrame:
    api = Api("https://fred.stlouisfed.org", pause=0.5)
    cols = {}
    for sid, name in series.items():
        code, text = api.get("/graph/fredgraph.csv", {"id": sid, "cosd": start}, raw=True)
        if code != 200 or not isinstance(text, str) or "," not in text:
            print(f"  !! FRED {sid}: HTTP {code} - skipped")
            continue
        df = pd.read_csv(io.StringIO(text))
        date_col = df.columns[0]
        s = pd.to_numeric(df[df.columns[1]], errors="coerce")      # '.' marks holidays
        s.index = pd.to_datetime(df[date_col])
        cols[sid] = s
        print(f"  FRED {sid:6s} {name}: {s.notna().sum():,} days, {s.index.min().date()} -> {s.index.max().date()}")
    if not cols:
        return pd.DataFrame()
    out = pd.DataFrame(cols).sort_index()
    out = out[out.index >= pd.Timestamp(start)]
    out.index.name = "date"
    out.to_csv(config.FRED_DIR / "daily_rates.csv")
    return out
