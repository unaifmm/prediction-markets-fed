"""
STEP 4 - Zoom in: the first minutes after CPI and payrolls, second by second.

Minute prices cannot say what happens inside the release minute, and
Polymarket stamps its minute prices a few seconds late (step 3). Trades can:
both venues record every trade to the second (Kalshi to the microsecond).

For every CPI and payrolls release since January 2025 we build, second by
second from 5 minutes before to 10 minutes after 08:30 New York:
  * Kalshi and Polymarket: the expected rate change over the next two decisions,
    where each outcome's probability is its latest TRADE price (it starts from the
    minute price 5 minutes before the release; a Polymarket NO trade at price q
    means YES = 1 - q);
  * S&P 500 futures and USD/JPY: log price from 1-second bars (S&P futures from
    September 2025, when the vendor's 1-second futures data starts).

Speed = share of the 5-minute reaction done after h seconds (same pooled
absorption curve as step 3), and the "half-life": seconds until half of it is done.
We also measure how long each venue waits for its first trade after 08:30:00.

Caveat: a trade price only changes when someone trades, and it bounces between
bid and ask. Quotes can move before trades, so trade-based speed is a slightly
pessimistic reading of how fast the venue's price adjusts.

Run:  python 04_seconds.py        Writes results/04_*.csv and figures/04_seconds.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, events, probs
from src.trades import load, trade_path, unix
from src.plots import GRID, INK_2, LABEL, MARKET, VENUE, save, set_style

SERIES = ("kalshi", "polymarket", "ES", "USDJPY")
SECS = np.arange(-60, 601)               # seconds around the release kept for the analysis
K0 = int(np.where(SECS == 0)[0][0])
ANCHOR = 300                             # the reaction is measured over 5 minutes
N_BOOT = 2000
def build() -> tuple[pd.DataFrame, dict, dict]:
    km, pm, mt = probs.markets()
    rel = events.load_events({k: events.RELEASES[k] for k in ("CPI", "Payrolls")})
    rel = rel[rel["time"] >= pd.Timestamp(config.SECONDS_FROM, tz=config.TZ)].reset_index(drop=True)
    info, paths = [], {s: [] for s in SERIES}
    counts = {"kalshi": [], "polymarket": []}
    for _, r in rel.iterrows():
        t0 = r["time"].tz_convert("UTC")
        T = int(t0.timestamp())
        tag = f"{r['kind']}_{r['time']:%Y-%m-%d}"
        grid = T + SECS
        kt, pt = load(tag)
        # prediction markets: next two decisions, same meetings on both venues
        total, n = {"kalshi": np.zeros(len(grid)), "polymarket": np.zeros(len(grid))}, 0
        for _, m in mt[mt["decision_time"] > t0 + pd.Timedelta(minutes=1)].head(2).iterrows():
            e = {v: trade_path(v, tag, m, km, pm, kt, pt, t0, grid)[0] for v in ("kalshi", "polymarket")}
            if any(x is None for x in e.values()):
                break
            total, n = {v: total[v] + e[v] for v in total}, n + 1
        for v in ("kalshi", "polymarket"):
            paths[v].append(total[v] if n else np.full(len(grid), np.nan))
        # fast markets: 1-second bars; the close of the bar [t-1, t) is the price at t
        for name in ("ES", "USDJPY"):
            f = config.DATA_LSE_1S / f"{name}_{tag}.parquet"
            d = pd.read_parquet(f) if f.exists() else pd.DataFrame()
            if len(d) > 100:
                s = pd.Series(np.log(d["close"].to_numpy()) * 1e4, index=unix(d["timestamp"]).astype("int64") + 1)
                paths[name].append(probs.on_grid({0: s}, grid, max_age=10 ** 9)[0].to_numpy())
            else:
                paths[name].append(np.full(len(grid), np.nan))
        # trading activity in the next meeting's markets
        nxt = probs.next_meeting(mt, t0)
        kk = kt[kt["ticker"].str.startswith(nxt["kalshi_event"] + "-")] if len(kt) else kt
        pe = pt[pt["event_id"].astype(str) == str(nxt["poly_event_id"])] if len(pt) else pt
        k_after = kk["ts"][kk["ts"] >= T] - T if len(kk) else pd.Series(dtype=float)
        p_after = pe["timestamp"][pe["timestamp"] >= T] - T if len(pe) else pd.Series(dtype=float)
        counts["kalshi"].append(np.histogram(kk["ts"] - T, bins=np.arange(-60, 601))[0] if len(kk) else np.zeros(660))
        counts["polymarket"].append(np.histogram(pe["timestamp"] - T, bins=np.arange(-60, 601))[0] if len(pe) else np.zeros(660))
        info.append({"release": tag, "kind": r["kind"], "year": r["time"].year, "meetings_used": n,
                     "kalshi_first_trade_s": k_after.min() if len(k_after) else np.nan,
                     "polymarket_first_trade_s": p_after.min() if len(p_after) else np.nan,
                     "kalshi_trades_first_minute": int((k_after < 60).sum()),
                     "polymarket_trades_first_minute": int((p_after < 60).sum())})
    return pd.DataFrame(info), {k: np.array(v) for k, v in paths.items()}, {k: np.array(v) for k, v in counts.items()}


def curve(x: np.ndarray, anchor: int = ANCHOR) -> np.ndarray:
    r = x[:, K0 + anchor] - x[:, K0]
    return (np.sign(r)[:, None] * (x - x[:, [K0]])).sum(axis=0) / np.abs(r).sum()


def half_life(c: np.ndarray, anchor: int = ANCHOR) -> float:
    return next((h for h in range(0, anchor + 1) if c[K0 + h] >= 0.5), np.nan)


def summarise_sample(label, paths, idx, series) -> list[dict]:
    rng = np.random.default_rng(11)
    draws = [rng.choice(idx, len(idx)) for _ in range(N_BOOT)]
    rows = []
    for s in series:
        c = curve(paths[s][idx])
        boot = np.array([curve(paths[s][d]) for d in draws])
        hl = np.array([half_life(b) for b in boot])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        rows.append({"sample": label, "series": s, "releases": len(idx),
                     **{f"share_{h}s": c[K0 + h] for h in (1, 5, 10, 30, 60, 120)},
                     "share_10s_low": lo[K0 + 10], "share_10s_high": hi[K0 + 10],
                     "share_60s_low": lo[K0 + 60], "share_60s_high": hi[K0 + 60],
                     "half_life_s": half_life(c), "half_life_low": np.nanpercentile(hl, 2.5),
                     "half_life_high": np.nanpercentile(hl, 97.5)})
    return rows


def main() -> None:
    info, paths, counts = build()
    has = {s: np.isfinite(paths[s]).all(axis=1) for s in SERIES}
    pm_fx = has["kalshi"] & has["polymarket"] & has["USDJPY"]
    four = pm_fx & has["ES"]
    y = info["year"].to_numpy()
    samples = [("all four markets (from Sep 2025)", np.where(four)[0], SERIES),
               ("2025-26, without S&P futures", np.where(pm_fx)[0], ("kalshi", "polymarket", "USDJPY")),
               ("2025", np.where(pm_fx & (y == 2025))[0], ("kalshi", "polymarket", "USDJPY")),
               ("2026", np.where(pm_fx & (y == 2026))[0], ("kalshi", "polymarket", "USDJPY"))]
    rows = []
    for label, idx, series in samples:
        rows += summarise_sample(label, paths, idx, series)
    out = pd.DataFrame(rows)
    out.round(3).to_csv(config.RESULTS / "04_seconds_summary.csv", index=False)
    info.to_csv(config.RESULTS / "04_first_trades.csv", index=False)

    print(f"{len(info)} CPI and payrolls releases since {config.SECONDS_FROM}")
    print("\nShare of the 5-minute reaction done after ... (95% CI); half-life = seconds to 50%")
    for label, idx, series in samples:
        print(f"\n  {label} (n={len(idx)})")
        for _, r in out[out["sample"] == label].iterrows():
            print(f"   {LABEL[r['series']]:16s} 5 s {r['share_5s']:4.2f}   10 s {r['share_10s']:4.2f} "
                  f"[{r['share_10s_low']:.2f}, {r['share_10s_high']:.2f}]   30 s {r['share_30s']:4.2f}   "
                  f"60 s {r['share_60s']:4.2f}   half-life {r['half_life_s']:5.0f} s "
                  f"[{r['half_life_low']:.0f}, {r['half_life_high']:.0f}]")
    print("\nFirst trade after 08:30:00 in the next meeting's markets (median seconds; trades in the first minute)")
    for yr, g in info.groupby("year"):
        print(f"   {yr}: Kalshi {g['kalshi_first_trade_s'].median():5.1f} s ({g['kalshi_trades_first_minute'].median():.0f} trades)   "
              f"Polymarket {g['polymarket_first_trade_s'].median():5.1f} s ({g['polymarket_trades_first_minute'].median():.0f} trades)")
    figure(paths, counts, info, samples)


def figure(paths, counts, info, samples) -> None:
    set_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    label, idx, series = samples[0]
    hh = np.arange(0, ANCHOR + 1)
    rng = np.random.default_rng(11)
    draws = [rng.choice(idx, len(idx)) for _ in range(N_BOOT)]
    for s in series:
        c = curve(paths[s][idx])
        color = VENUE.get(s, MARKET.get(s))
        a1.plot(hh, c[K0:K0 + ANCHOR + 1], color=color, label=LABEL[s],
                linewidth=2 if s in VENUE else 1.3, linestyle="-" if s in VENUE else ":")
        if s in VENUE:
            lo, hi = np.percentile([curve(paths[s][d]) for d in draws], [2.5, 97.5], axis=0)
            a1.fill_between(hh, lo[K0:K0 + ANCHOR + 1], hi[K0:K0 + ANCHOR + 1], color=color, alpha=0.12, linewidth=0)
    a1.axhline(0.5, color=INK_2, linewidth=0.7, linestyle="--")
    a1.set_xscale("symlog", linthresh=10)
    a1.set_xlim(0, ANCHOR)
    a1.set_xticks([0, 1, 5, 10, 30, 60, 120, 300], ["0", "1", "5", "10", "30", "60", "120", "300"])
    a1.set_xlabel("seconds after the release (08:30:00 New York)")
    a1.set_ylabel("share of the 5-minute reaction done")
    a1.set_title(f"Second by second ({len(idx)} releases, Sep 2025 - Oct 2026)")
    a1.legend(loc="upper left")

    k = np.arange(0, 120)
    for v in ("kalshi", "polymarket"):
        for yr, ls in ((2025, ":"), (2026, "-")):
            sel = (info["year"] == yr).to_numpy()
            c = counts[v][sel].mean(axis=0)
            base = c[:60].mean() or 1e-9
            sm = pd.Series(c[60:180]).rolling(5, min_periods=1).mean().to_numpy() / base
            a2.plot(k, sm, color=VENUE[v], linestyle=ls, label=f"{LABEL[v]} {yr}")
    a2.set_yscale("log")
    a2.set_xlabel("seconds after the release")
    a2.set_ylabel("trades per second,\nx the minute before (5 s average)")
    a2.set_title("Trading wakes up faster in 2026")
    a2.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "04_seconds.png")


if __name__ == "__main__":
    main()
