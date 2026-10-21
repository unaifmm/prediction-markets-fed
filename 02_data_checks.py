"""
STEP 2 - Can we trust the data? Four checks before any research question.

  A. Settlement   Do both venues settle every meeting on the outcome that really
                  happened? (cross-checked with the effective fed funds rate, FRED)
  B. Coherence    Do each venue's outcome prices add up to about 1, as probabilities must?
  C. Agreement    Do Kalshi and Polymarket tell the same story, hour by hour?
  D. Clocks       Around CPI and payrolls, do the prediction markets jump in the
                  release minute (08:30 New York), like S&P 500 futures and USD/JPY?
                  A jump in the wrong minute would mean a timestamp problem.

Run:  python 02_data_checks.py      (needs the data from 01_download_data.py)
Writes results/02_*.csv and figures/02_*.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config, events, probs
from src.lse import load_lse
from src.plots import INK_2, LABEL, MARKET, VENUE, GRID, save, set_style

HOURS_BEFORE = 60 * 24          # checks B and C use the last 60 days before each decision
SPAN = (-30, 60)                # check D: minutes around the release
MAX_AGE = 3 * 24 * 3600         # a quiet outcome (e.g. a 1% tail) keeps its last price for up to 3 days


def check_settlement(km, pm, mt) -> pd.DataFrame:
    print("\nA. Settlement: what each venue paid out vs the real change in the fed funds rate")
    kal = km[km["result"] == "yes"].groupby("event")["bp"].first()
    pol = pm[pm["final_yes"] > 0.99].groupby("event")["bp"].first()
    ff = pd.read_csv(config.FRED_DIR / "daily_rates.csv", parse_dates=["date"]).set_index("date")["DFF"].dropna()
    rows = []
    for _, m in mt.iterrows():
        day = m["decision_time"].tz_convert(config.TZ).tz_localize(None).normalize()
        before, after = ff[:day - pd.Timedelta(days=1)], ff[day + pd.Timedelta(days=1):]
        change = round((after.iloc[0] - before.iloc[-1]) * 100) if len(before) and len(after) else np.nan
        rows.append({"meeting": m["meeting"], "kalshi_paid_bp": kal.get(m["kalshi_event"], np.nan),
                     "polymarket_paid_bp": pol.get(m["poly_event"], np.nan), "fed_funds_change_bp": change})
    df = pd.DataFrame(rows)
    done = df.dropna(subset=["kalshi_paid_bp", "polymarket_paid_bp"])
    same = (done["kalshi_paid_bp"] == done["polymarket_paid_bp"]).sum()
    real = ((done["kalshi_paid_bp"] - done["fed_funds_change_bp"]).abs() <= 5).sum()
    print(f"   settled meetings: {len(done)}; venues agree on {same}; "
          f"both match the fed funds change (within 5 bp) on {real}")
    print("   decisions: " + ", ".join(f"{int(v):+d} bp x{n}" for v, n in
                                       done["kalshi_paid_bp"].value_counts().sort_index().items()))
    df.to_csv(config.RESULTS / "02_settlement.csv", index=False)
    return df


def hourly_panel(km, pm, mt) -> pd.DataFrame:
    rows = []
    for _, m in mt.iterrows():
        t1 = int(m["decision_time"].timestamp())
        grid = np.arange(t1 - HOURS_BEFORE * 3600, t1 - 1800, 3600)
        grid = grid - grid % 3600
        for venue in ("kalshi", "polymarket"):
            s = probs.hourly_series(venue, m, km, pm)
            if not s:
                continue
            summ = probs.summarise(probs.on_grid(s, grid, max_age=48 * 3600))
            summ["meeting"], summ["venue"], summ["ts"] = m["meeting"], venue, grid
            rows.append(summ)
    return pd.concat(rows, ignore_index=True)


def check_coherence(panel) -> pd.DataFrame:
    print("\nB. Coherence: do the outcome prices add up to 1? (hourly, last 60 days before each decision)")
    d = panel.dropna(subset=["total"])
    q = d.groupby("venue")["total"].quantile([0.05, 0.5, 0.95]).unstack()
    for v, r in q.iterrows():
        print(f"   {LABEL[v]:10s} median {r[0.5]:.3f}   90% of hours between {r[0.05]:.3f} and {r[0.95]:.3f}")
    out = d.groupby(["meeting", "venue"])["total"].median().unstack()
    out.to_csv(config.RESULTS / "02_probability_sums.csv")
    return out


def check_agreement(panel) -> pd.DataFrame:
    print("\nC. Agreement: Kalshi vs Polymarket, same meeting, same hour")
    w = panel.pivot_table(index=["meeting", "ts"], columns="venue", values=["exp_bp", "p_hold"]).dropna()
    gap = w[("exp_bp", "kalshi")] - w[("exp_bp", "polymarket")]
    hold = w[("p_hold", "kalshi")] - w[("p_hold", "polymarket")]
    corr = np.corrcoef(w[("exp_bp", "kalshi")], w[("exp_bp", "polymarket")])[0, 1]
    print(f"   {len(w):,} common hours, {w.index.get_level_values(0).nunique()} meetings")
    print(f"   expected rate change: correlation {corr:.3f}; gap median {gap.abs().median():.2f} bp, "
          f"95% of hours below {gap.abs().quantile(0.95):.2f} bp")
    print(f"   probability of no change: gap median {hold.abs().median() * 100:.1f} points, "
          f"95% of hours below {hold.abs().quantile(0.95) * 100:.1f} points")
    by = pd.DataFrame({"hours": gap.groupby(level=0).size(),
                       "median_gap_bp": gap.abs().groupby(level=0).median(),
                       "p95_gap_bp": gap.abs().groupby(level=0).quantile(0.95),
                       "mean_gap_bp": gap.groupby(level=0).mean()})
    by.round(3).to_csv(config.RESULTS / "02_venue_agreement.csv")
    return w


def release_minutes(km, pm, mt) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Absolute minute-by-minute change of each venue's expected rate change (next meeting)
    and of ES / USD/JPY returns, around every CPI and payrolls release."""
    rel = events.load_events({k: events.RELEASES[k] for k in ("CPI", "Payrolls")})
    rel = rel[rel["time"] >= pd.Timestamp(config.START, tz=config.TZ)]
    px = load_lse(["ES", "USDJPY"], how="outer")
    ret = np.log(px).diff().abs() * 1e4                          # |1-minute return| in bp
    ks = np.arange(SPAN[0], SPAN[1] + 1)
    rows, peaks = [], []
    for _, r in rel.iterrows():
        t0 = r["time"].tz_convert("UTC")
        tag = f"{r['kind']}_{r['time']:%Y-%m-%d}"
        meet = probs.next_meeting(mt, t0)
        if meet is None:
            continue
        row = {"release": tag, "kind": r["kind"]}
        grid = int(t0.timestamp()) + np.arange(SPAN[0], SPAN[1] + 2) * 60   # minute boundaries
        for venue in ("kalshi", "polymarket"):
            s = probs.minute_series(venue, tag, meet, km, pm, start_ts=int((t0 - config.WINDOW_BEFORE).timestamp()))
            e = probs.summarise(probs.on_grid(s, grid, max_age=MAX_AGE))["exp_bp"].to_numpy() if s else np.full(len(grid), np.nan)
            move = np.abs(np.diff(e))                             # move during minute k = [t0+k, t0+k+1)
            rows += [{"release": tag, "kind": r["kind"], "series": venue, "k": k, "move": v} for k, v in zip(ks, move)]
            row[venue] = ks[np.nanargmax(move)] if np.isfinite(move).any() and np.nanmax(move) > 0 else np.nan
        for name in ("ES", "USDJPY"):
            idx = r["time"] + pd.to_timedelta(ks, unit="min")
            v = ret[name].reindex(idx).to_numpy()
            rows += [{"release": tag, "kind": r["kind"], "series": name, "k": k, "move": x} for k, x in zip(ks, v)]
            row[name] = ks[np.nanargmax(v)] if np.isfinite(v).any() else np.nan
        peaks.append(row)
    return pd.DataFrame(rows), pd.DataFrame(peaks)


def check_clocks(moves, peaks) -> pd.DataFrame:
    print("\nD. Clocks: in which minute is the biggest move, around CPI and payrolls? (0 = 08:30-08:31 New York)")
    out = []
    for s in ("ES", "USDJPY", "kalshi", "polymarket"):
        p = peaks[s].dropna()
        out.append({"series": LABEL[s], "releases": len(p), "in_release_minute_%": 100 * (p == 0).mean(),
                    "within_2_minutes_%": 100 * p.between(0, 2).mean(),
                    "before_release_%": 100 * (p < 0).mean()})
        print(f"   {LABEL[s]:16s} {len(p):3d} releases: biggest move in the release minute {100 * (p == 0).mean():5.1f}%, "
              f"within 2 minutes {100 * p.between(0, 2).mean():5.1f}%, before it {100 * (p < 0).mean():4.1f}%")
    df = pd.DataFrame(out)
    df.round(1).to_csv(config.RESULTS / "02_timing_check.csv", index=False)
    peaks.to_csv(config.RESULTS / "02_biggest_move_minute.csv", index=False)
    return df


def figures(panel, wide, moves) -> None:
    set_style()
    # Figure 1: the venues agree
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    show = "2026-09-16"
    for v in ("kalshi", "polymarket"):
        d = panel[(panel["meeting"] == show) & (panel["venue"] == v)]
        a1.plot(pd.to_datetime(d["ts"], unit="s"), d["exp_bp"], color=VENUE[v], label=LABEL[v],
                linewidth=1.4 if v == "kalshi" else 1.0)
    a1.set_title(f"Expected rate change for the {show} decision, hourly")
    a1.set_ylabel("bp (probability-weighted)")
    a1.legend(loc="upper left")
    x, y = wide[("exp_bp", "kalshi")], wide[("exp_bp", "polymarket")]
    a2.scatter(y, x, s=3, alpha=0.25, color=VENUE["kalshi"], edgecolors="none")
    lim = [min(x.min(), y.min()) - 2, max(x.max(), y.max()) + 2]
    a2.plot(lim, lim, color=INK_2, linewidth=0.8, linestyle="--")
    a2.set_xlim(lim), a2.set_ylim(lim)
    a2.set_xlabel("Polymarket (bp)"), a2.set_ylabel("Kalshi (bp)")
    a2.set_title(f"All meetings, every hour (corr {np.corrcoef(x, y)[0, 1]:.3f})")
    fig.suptitle("Kalshi and Polymarket tell the same story", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "02_venues_agree.png")

    # Figure 2: everyone jumps in the release minute
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for s in ("ES", "USDJPY", "kalshi", "polymarket"):
        d = moves[moves["series"] == s].groupby("k")["move"].mean()
        base = d.loc[SPAN[0]:-6].mean()
        color = MARKET.get(s, VENUE.get(s))
        ax.plot(d.index, d / base, color=color, label=LABEL[s], linewidth=1.8 if s in VENUE else 1.2,
                linestyle="-" if s in VENUE else ":")
    ax.axvline(0, color=GRID, linewidth=6, zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("minutes after the release (0 = 08:30-08:31 New York)")
    ax.set_ylabel("average absolute move,\nx the pre-release average (log)")
    ax.set_title("Around CPI and payrolls every market moves in the release minute: the clocks are right")
    ax.legend(ncol=4, loc="upper right")
    save(fig, "02_timing_check.png")


def main() -> None:
    km, pm, mt = probs.markets()
    print(f"{len(mt)} meetings, {len(km)} Kalshi markets, {len(pm)} Polymarket markets")
    check_settlement(km, pm, mt)
    panel = hourly_panel(km, pm, mt)
    check_coherence(panel)
    wide = check_agreement(panel)
    moves, peaks = release_minutes(km, pm, mt)
    check_clocks(moves, peaks)
    figures(panel, wide, moves)


if __name__ == "__main__":
    main()
