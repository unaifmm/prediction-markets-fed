"""
Macro announcements: when they happened, how surprising they were, and the
minutes of data around them.

DATA-QUALITY NOTE - the vendor's timestamps are an hour late for most of the
history. In the London Strategic Edge calendar, entries back-filled for
2016-2025 are stamped ONE HOUR LATER than the true release (payrolls at 09:30
New York instead of 08:30, the FOMC at 15:00 instead of 14:00), in winter and
summer alike, while entries recorded live in 2026 are right. The prices prove
it: the biggest 1-minute NQ move within +/-90 minutes falls exactly on the
official release minute on about half of CPI and payroll days, and on the
vendor's minute on fewer than 1 in 10 (08_event_study.py, part A).

So we take the DATE from the calendar and the TIME from the official schedule,
which did not change during the sample:
    CPI (Bureau of Labor Statistics)           08:30 New York
    Employment Situation / payrolls (BLS)      08:30
    ISM Manufacturing PMI                      10:00
    FOMC rate decision (statement)             14:00
The two unscheduled FOMC cuts (3 and 15 March 2020) are left out.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

# kind -> (event name in the vendor calendar, official release time in New York)
RELEASES = {
    "CPI": (r"^Inflation Rate MoM( [A-Z]{3})?$", "08:30"),
    "Payrolls": (r"^Non Farm Payrolls( [A-Z]{3})?$", "08:30"),
    "ISM": (r"^ISM Manufacturing PMI( [A-Z]{3})?$", "10:00"),
    "FOMC": (r"^Fed Interest Rate Decision$", "14:00"),
}


def parse_number(x) -> float:
    """'0.3%' -> 0.3, '172K' -> 172, '-23K' -> -23, '48.7' -> 48.7, None -> NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    m = re.search(r"-?\d+(\.\d+)?", str(x).replace(",", ""))
    return float(m.group()) if m else np.nan


def load_events(kinds: dict = RELEASES, path=None) -> pd.DataFrame:
    """One row per release: kind, date, official time (New York), the vendor's
    time, actual, consensus and the surprise (actual - consensus), also
    standardised by the spread of that release's surprises (surprise_z)."""
    cal = pd.read_parquet(path or config.ECON_CALENDAR)
    cal["vendor_time"] = pd.to_datetime(cal["datetime"], utc=True).dt.tz_convert(config.TZ)
    rows = []
    for kind, (pattern, clock) in kinds.items():
        sel = cal[cal["event"].str.match(pattern)].sort_values("vendor_time")
        for date, g in sel.groupby("date"):
            first = g.iloc[0]
            rows.append({"kind": kind, "date": pd.Timestamp(date),
                         "time": pd.Timestamp(f"{date} {clock}", tz=config.TZ),
                         "vendor_time": first["vendor_time"], "event": first["event"],
                         "actual": parse_number(first["actual"]),
                         "consensus": parse_number(first["consensus"])})
    ev = pd.DataFrame(rows)
    offset = (ev["vendor_time"] - ev["time"]).dt.total_seconds() / 60
    ev = ev[offset.isin([0, 60])]                        # drops the unscheduled FOMC moves
    ev = ev[ev["actual"].notna()]                        # not released yet
    ev["surprise"] = ev["actual"] - ev["consensus"]
    ev["surprise_z"] = ev["surprise"] / ev.groupby("kind")["surprise"].transform("std")
    return ev.sort_values("time").reset_index(drop=True)


def stack_windows(returns: pd.DataFrame, starts, before: int = 5, after: int = 20,
                  freq: str = "1min") -> pd.DataFrame:
    """Cut a window of bars around each start time and lay the windows end to
    end on an artificial clock, one day apart, so that:
      * clock-based lags never reach from one window into the next, and
      * the same event can appear twice (bootstrap resampling).
    Bars run from `before` bars before each start to `after` bars after it.
    Adds a column 'k' = bar number relative to the start (0 = the start bar)."""
    step = pd.Timedelta(freq)
    offsets = np.arange(-before, after)
    base = pd.Timestamp("2000-01-03", tz=config.TZ)
    parts = []
    for i, t0 in enumerate(starts):
        w = returns.reindex(t0 + offsets * step)
        w.index = base + pd.Timedelta(days=i) + offsets * step
        w["k"] = offsets
        parts.append(w)
    return pd.concat(parts) if parts else returns.iloc[0:0].assign(k=[])
