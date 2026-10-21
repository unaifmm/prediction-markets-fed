"""
STEP 6 - Research question 2: Kalshi vs Polymarket.

  A. Who moves first after CPI and payrolls? Second by second from trades (as in
     step 4), 2025-26:
       * lead-lag asymmetry  corr(dK_t, dP_t+1) - corr(dP_t, dK_t+1)  on 2-, 5- and
         10-second changes during the first 5 minutes (> 0 = Kalshi first). As in
         project 1's staleness trap, a venue that trades less often keeps an old
         price for longer and LOOKS like a follower even if it is not; Polymarket
         trades about twice as often as Kalshi at 08:30. Fix: use only intervals in
         which BOTH venues traded. 95% CI: bootstrap over releases.
       * half-life of each venue's 5-minute reaction, release by release (releases
         where both moved at least 1 bp), and the paired difference.
  B. Do prices for the same outcome ever differ by more than the cost of trading
     the gap? Hourly, last 60 days before each decision, outcomes priced 3-97% on
     both venues. Buy YES on the cheaper venue and NO on the dearer one: together
     they pay exactly $1 at the decision (both venues settled all 28 past meetings
     identically, step 2), so the profit is locked in:
         Kalshi YES at ask + Polymarket NO   or   Kalshi NO at (1 - bid) + Polymarket YES
     Main sample: liquid markets only - meetings where Polymarket traded at least $10
     million (from September 2024) and hours when Kalshi's spread is 5 cents or less.
     Earlier Polymarket markets were tiny and their prices often were not tradable
     (e.g. stuck at exactly 0.50, an empty order book); they are shown separately.
     Costs: Kalshi's quoted spread (best bid/ask at the hour), an assumed Polymarket
     half-spread of 0.5 cents (1 cent as robustness; its price history has no
     quotes), and both venues' taker fees. The money is tied up until the decision,
     so the return is also annualised and compared with the 3-month T-bill.

Run:  python 06_venues.py        Writes results/06_*.csv and figures/06_venues.png
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, events, probs
from src.kalshi import file_name
from src.plots import AQUA, BLUE, INK_2, VENUE, save, set_style
from src.trades import load, trade_path

SECS = np.arange(-300, 601)
K0 = int(np.where(SECS == 0)[0][0])
N_BOOT = 1000
POLY_HALF_SPREAD = 0.005
MIN_POLY_VOLUME = 10e6              # liquid meetings: at least $10 million traded on Polymarket
MAX_KALSHI_SPREAD = 0.05


# ---------------------------------------------------------------------------
# A. Who moves first
# ---------------------------------------------------------------------------
def build_paths(km, pm, mt) -> list[dict]:
    rel = events.load_events({k: events.RELEASES[k] for k in ("CPI", "Payrolls")})
    rel = rel[rel["time"] >= pd.Timestamp(config.SECONDS_FROM, tz=config.TZ)].reset_index(drop=True)
    out = []
    for _, r in rel.iterrows():
        t0 = r["time"].tz_convert("UTC")
        tag = f"{r['kind']}_{r['time']:%Y-%m-%d}"
        grid = int(t0.timestamp()) + SECS
        kt, pt = load(tag)
        path = {"kalshi": np.zeros(len(grid)), "polymarket": np.zeros(len(grid))}
        times = {"kalshi": [], "polymarket": []}
        n = 0
        for _, m in mt[mt["decision_time"] > t0 + pd.Timedelta(minutes=1)].head(2).iterrows():
            res = {v: trade_path(v, tag, m, km, pm, kt, pt, t0, grid) for v in path}
            if any(x[0] is None for x in res.values()):
                break
            for v in path:
                path[v] += res[v][0]
                times[v] += list(res[v][1])
            n += 1
        if n:
            out.append({"release": tag, "year": r["time"].year, "T": int(t0.timestamp()),
                        "K": path["kalshi"], "P": path["polymarket"],
                        "kt": np.array(times["kalshi"]), "pt": np.array(times["polymarket"])})
    return out


def asymmetry(data, sel, step: int, both: bool, span: int = 300) -> tuple[float, int]:
    edges = np.arange(0, span + step, step)
    x1, y1, x2, y2 = [], [], [], []
    for i in sel:
        d = data[i]
        dk, dp = np.diff(d["K"][K0 + edges]), np.diff(d["P"][K0 + edges])
        k_traded = np.histogram(d["kt"] - d["T"], bins=edges)[0] > 0
        p_traded = np.histogram(d["pt"] - d["T"], bins=edges)[0] > 0
        ok = (k_traded & p_traded) if both else np.ones(len(dk), bool)
        for t in np.where(ok[:-1])[0]:
            x1.append(dk[t]), y1.append(dp[t + 1]), x2.append(dp[t]), y2.append(dk[t + 1])
    if len(x1) < 10:
        return np.nan, len(x1)
    return np.corrcoef(x1, y1)[0, 1] - np.corrcoef(x2, y2)[0, 1], len(x1)


def half_life(x: np.ndarray, anchor: int = 300) -> float:
    r = x[K0 + anchor] - x[K0]
    if abs(r) < 1:                                  # a reaction under 1 bp has no meaningful half-life
        return np.nan
    s = (x[K0:K0 + anchor + 1] - x[K0]) / r
    i = int(np.argmax(s >= 0.5))
    return float(i) if s[i] >= 0.5 else np.nan


def who_first(data) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.arange(len(data))
    rng = np.random.default_rng(3)
    rows = []
    for step in (2, 5, 10):
        for both in (True, False):
            a, n = asymmetry(data, idx, step, both)
            boot = [asymmetry(data, rng.choice(idx, len(idx)), step, both)[0] for _ in range(N_BOOT)]
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            rows.append({"interval_s": step, "only_when_both_traded": both, "asymmetry_kalshi_first": a,
                         "low": lo, "high": hi, "pairs": n})
    asym = pd.DataFrame(rows)
    hl = pd.DataFrame([{"release": d["release"], "year": d["year"], "kalshi_half_life_s": half_life(d["K"]),
                        "polymarket_half_life_s": half_life(d["P"]),
                        "kalshi_5min_bp": d["K"][K0 + 300] - d["K"][K0],
                        "polymarket_5min_bp": d["P"][K0 + 300] - d["P"][K0]} for d in data])
    return asym, hl


# ---------------------------------------------------------------------------
# B. Price gaps vs costs
# ---------------------------------------------------------------------------
def hourly_pairs(km, pm, mt) -> pd.DataFrame:
    rows = []
    for _, m in mt.iterrows():
        t1 = int(m["decision_time"].timestamp())
        grid = np.arange(t1 - 60 * 86400, t1 - 3600, 3600)
        grid = grid - grid % 3600
        common = set(km[km["event"] == m["kalshi_event"]]["bp"].dropna()) & set(pm[pm["event"] == m["poly_event"]]["bp"].dropna())
        for bp in sorted(common):
            k_m = km[(km["event"] == m["kalshi_event"]) & (km["bp"] == bp)].iloc[0]
            p_m = pm[(pm["event"] == m["poly_event"]) & (pm["bp"] == bp)]
            if len(p_m) != 1:
                continue
            p_m = p_m.iloc[0]
            fk = config.KALSHI_DIR / "hourly" / f"{file_name(k_m['ticker'])}.parquet"
            fp = config.POLY_DIR / "hourly" / f"{p_m['market_id']}.parquet"
            if not (fk.exists() and fp.exists()):
                continue
            k, p = pd.read_parquet(fk), pd.read_parquet(fp)
            if not (len(k) and len(p)):
                continue
            g = probs.on_grid({"bid": pd.Series(k["yes_bid_close"].to_numpy(), index=k["ts"].to_numpy()),
                               "ask": pd.Series(k["yes_ask_close"].to_numpy(), index=k["ts"].to_numpy()),
                               "poly": pd.Series(p["p"].to_numpy(), index=(p["ts"] // 60 * 60).to_numpy())},
                              grid, max_age=3600)
            d = g.assign(ts=grid).dropna()
            d = d[d["poly"].between(0.03, 0.97) & (d["bid"] > 0) & (d["ask"] >= d["bid"])]
            if len(d):
                rows.append(d.assign(meeting=m["meeting"], bp=bp, poly_fee_rate=p_m["fee_rate"],
                                     poly_volume=m["poly_volume"], days_to_decision=(t1 - d["ts"]) / 86400))
    return pd.concat(rows, ignore_index=True)


def locked_in(d: pd.DataFrame, half_spread: float) -> pd.DataFrame:
    kf = lambda q: 0.07 * q * (1 - q)
    pf = lambda q, r: r * q * (1 - q)
    # (i) YES on Kalshi at the ask + NO on Polymarket;  (ii) NO on Kalshi (1 - bid) + YES on Polymarket
    cost1 = d["ask"] + (1 - d["poly"] + half_spread)
    gain1 = 1 - cost1 - kf(d["ask"]) - pf(d["poly"] - half_spread, d["poly_fee_rate"])
    cost2 = (1 - d["bid"]) + d["poly"] + half_spread
    gain2 = 1 - cost2 - kf(d["bid"]) - pf(d["poly"] + half_spread, d["poly_fee_rate"])
    out = d.assign(profit=np.maximum(gain1, gain2), capital=np.where(gain1 >= gain2, cost1, cost2))
    out["annualised"] = out["profit"] / out["capital"] * 365 / out["days_to_decision"].clip(lower=1)
    f = config.FRED_DIR / "daily_rates.csv"
    if f.exists():
        tbill = pd.read_csv(f, parse_dates=["date"]).set_index("date")["DTB3"].ffill() / 100
        out["tbill"] = tbill.reindex(pd.to_datetime(out["ts"], unit="s").dt.normalize(), method="ffill").to_numpy()
    else:
        out["tbill"] = np.nan
    out["year"] = pd.to_datetime(out["ts"], unit="s").dt.year
    out["mid_gap_c"] = 100 * (out["poly"] - (out["bid"] + out["ask"]) / 2).abs()
    return out


def gap_summary(e: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for key, g in [("all", e)] + list(e.groupby("year")):
        pos = g[g["profit"] > 0]
        rows.append({"assumption": label, "period": key, "outcome_hours": len(g),
                     "mean_price_gap_c": g["mid_gap_c"].mean(),
                     "hours_with_locked_in_profit_%": 100 * len(pos) / len(g),
                     "median_profit_c": 100 * pos["profit"].median() if len(pos) else np.nan,
                     "p90_profit_c": 100 * pos["profit"].quantile(0.9) if len(pos) else np.nan,
                     "share_beating_tbill_%": 100 * (pos["annualised"] > pos["tbill"]).mean() if len(pos) else np.nan})
    return rows


def main() -> None:
    km, pm, mt = probs.markets()
    pm["fee_rate"] = pm["fee_schedule"].map(lambda s: json.loads(s).get("rate", 0.0) if isinstance(s, str) else 0.0)

    data = build_paths(km, pm, mt)
    asym, hl = who_first(data)
    asym.round(4).to_csv(config.RESULTS / "06_lead_lag.csv", index=False)
    hl.round(2).to_csv(config.RESULTS / "06_half_lives.csv", index=False)
    print(f"A. Who moves first after CPI and payrolls ({len(data)} releases since {config.SECONDS_FROM})")
    print("   lead-lag asymmetry (> 0 = Kalshi first), 95% CI:")
    for _, r in asym.iterrows():
        print(f"     {r['interval_s']:2d}-second changes, {'both traded' if r['only_when_both_traded'] else 'all intervals':13s} "
              f"{r['asymmetry_kalshi_first']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}]   ({r['pairs']} pairs)")
    both = hl.dropna(subset=["kalshi_half_life_s", "polymarket_half_life_s"])
    diff = both["polymarket_half_life_s"] - both["kalshi_half_life_s"]
    rng = np.random.default_rng(4)
    ci = np.percentile([np.median(rng.choice(diff.to_numpy(), len(diff))) for _ in range(5000)], [2.5, 97.5])
    print(f"   half-life, {len(both)} releases where both moved 1 bp+: Kalshi median {both['kalshi_half_life_s'].median():.0f} s, "
          f"Polymarket {both['polymarket_half_life_s'].median():.0f} s; Polymarket minus Kalshi median {diff.median():+.0f} s "
          f"[{ci[0]:+.0f}, {ci[1]:+.0f}]; Polymarket faster in {(diff < 0).sum()}, Kalshi in {(diff > 0).sum()}")
    print(f"   size of the 5-minute reaction: correlation across venues "
          f"{np.corrcoef(both['kalshi_5min_bp'], both['polymarket_5min_bp'])[0, 1]:.2f}")

    pairs = hourly_pairs(km, pm, mt)
    liquid = pairs[(pairs["poly_volume"] >= MIN_POLY_VOLUME) & (pairs["ask"] - pairs["bid"] <= MAX_KALSHI_SPREAD)]
    rows = (gap_summary(locked_in(liquid, POLY_HALF_SPREAD), "liquid, Polymarket half-spread 0.5 c")
            + gap_summary(locked_in(liquid, 0.01), "liquid, Polymarket half-spread 1 c")
            + gap_summary(locked_in(pairs, POLY_HALF_SPREAD), "all meetings (incl. illiquid 2023-24)"))
    gaps = pd.DataFrame(rows)
    gaps.round(2).to_csv(config.RESULTS / "06_price_gaps.csv", index=False)
    print(f"\nB. Same outcome, two prices: {len(liquid):,} liquid outcome-hours, {liquid['meeting'].nunique()} meetings "
          f"(all: {len(pairs):,} hours, {pairs['meeting'].nunique()} meetings); outcomes priced 3-97%")
    for _, r in gaps.iterrows():
        print(f"   {r['assumption']:38s} {str(r['period']):4s}  mean gap {r['mean_price_gap_c']:4.2f} c   "
              f"locked-in profit after costs in {r['hours_with_locked_in_profit_%']:4.1f}% of hours, "
              f"median {r['median_profit_c']:4.2f} c (90th pct {r['p90_profit_c']:4.2f} c), "
              f"{r['share_beating_tbill_%']:3.0f}% of them beat the T-bill")
    figure(both, asym, gaps)


def figure(both, asym, gaps) -> None:
    set_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    lim = [0, max(both["kalshi_half_life_s"].max(), both["polymarket_half_life_s"].max()) * 1.1]
    a1.scatter(both["kalshi_half_life_s"], both["polymarket_half_life_s"], color=BLUE, s=36, alpha=0.8, edgecolors="none")
    a1.plot(lim, lim, color=INK_2, linewidth=0.8, linestyle="--")
    a1.set_xlim(lim), a1.set_ylim(lim)
    a1.set_xlabel("Kalshi: seconds to half of the 5-minute reaction")
    a1.set_ylabel("Polymarket: seconds to half")
    r = asym[(asym["interval_s"] == 5) & asym["only_when_both_traded"]].iloc[0]
    a1.text(0.03, 0.97, f"below the line = Polymarket faster\n5-s lead-lag asymmetry {r['asymmetry_kalshi_first']:+.2f} "
            f"[{r['low']:+.2f}, {r['high']:+.2f}]", transform=a1.transAxes, va="top", fontsize=8.5, color=INK_2)
    a1.set_title(f"No leader: release by release (n={len(both)})")

    g = gaps[(gaps["assumption"] == "liquid, Polymarket half-spread 0.5 c") & (gaps["period"] != "all")]
    x = np.arange(len(g))
    a2.bar(x, g["hours_with_locked_in_profit_%"], width=0.55, color=VENUE["polymarket"])
    for xi, (_, r) in zip(x, g.iterrows()):
        a2.text(xi, r["hours_with_locked_in_profit_%"] + 0.6,
                f"median {r['median_profit_c']:.2f} c\naverage gap {r['mean_price_gap_c']:.1f} c",
                ha="center", va="bottom", fontsize=8.5, color=INK_2)
    a2.set_xticks(x, [str(p) for p in g["period"]])
    a2.set_ylim(0, g["hours_with_locked_in_profit_%"].max() * 1.35)
    a2.set_ylabel("% of hours with a locked-in profit\nafter spread and fees")
    a2.set_title("The gap between the venues is closing (liquid markets)")
    fig.tight_layout()
    save(fig, "06_venues.png")


if __name__ == "__main__":
    main()
