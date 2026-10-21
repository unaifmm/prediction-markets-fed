"""
STEP 7 - Research question 3: how good are the forecasts, and do they move with
the bond market?

  A. Accuracy. For each of the 28 decided meetings since May 2023, the
     probabilities each venue gave 1 day, 1 week, 1 month and 2 months before the
     decision (hourly prices; the outcomes rescaled to add up to 1):
       * did the favourite outcome happen?   * probability given to what happened
       * Brier score = sum over outcomes of (probability - 1 if it happened else 0)^2;
         0 = perfect, lower is better.
     Benchmarks with no market information: "the Fed holds" and "the Fed repeats
     its last decision".
     Calibration: of all outcomes priced at about 10% (or 30%, 70%, ...), how many
     happened? Well-calibrated forecasts lie on the 45-degree line. Pooled over
     both venues; 95% Wilson intervals. With 28 meetings this is a rough check.
  B. Bond market. Each business day at 16:00 New York: the expected rate change
     over the next two decisions (only those both venues list), day-on-day change,
     against the day's change in the 3- and 6-month T-bill and 1-, 2- and 10-year
     Treasury yields (FRED). Correlation, regression slope (Newey-West t), and
     whether one moves first from one day to the next.

Run:  python 07_accuracy_bonds.py      Writes results/07_*.csv and figures/07_accuracy_bonds.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import config, probs
from src.plots import GRID, INK_2, LABEL, VENUE, save, set_style

HORIZONS = {"1 day": 1, "1 week": 7, "1 month": 30, "2 months": 60}
BUCKETS = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001]
YIELDS = {"DTB3": "3-month bill", "DTB6": "6-month bill", "DGS1": "1-year", "DGS2": "2-year", "DGS10": "10-year"}


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


# ---------------------------------------------------------------------------
# A. Accuracy
# ---------------------------------------------------------------------------
def accuracy(km, pm, mt) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    actual = km[km["result"] == "yes"].groupby("event")["bp"].first()
    done = mt.assign(actual=mt["kalshi_event"].map(actual)).dropna(subset=["actual"]).reset_index(drop=True)
    rows, points = [], []
    for _, m in done.iterrows():
        t1 = int(m["decision_time"].timestamp())
        for venue in ("kalshi", "polymarket"):
            s = probs.hourly_series(venue, m, km, pm)
            if not s:
                continue
            for h, days in HORIZONS.items():
                p = probs.on_grid(s, np.array([t1 - days * 86400]), max_age=72 * 3600).iloc[0]
                if p.isna().all() or p.fillna(0).sum() < 0.8:
                    continue                                   # market not open yet, or no prices
                p = p.fillna(0) / p.fillna(0).sum()
                truth = (p.index.to_numpy() == m["actual"]).astype(float)
                rows.append({"meeting": m["meeting"], "venue": venue, "horizon": h, "actual_bp": m["actual"],
                             "favourite_bp": p.idxmax(), "favourite_p": p.max(),
                             "p_of_what_happened": p.get(m["actual"], 0.0),
                             "brier": ((p.to_numpy() - truth) ** 2).sum(), "favourite_right": p.idxmax() == m["actual"]})
                points += [{"venue": venue, "horizon": h, "p": pk, "happened": float(bp == m["actual"])}
                           for bp, pk in p.items()]
    per = pd.DataFrame(rows)
    pts = pd.DataFrame(points)
    a = done["actual"].to_numpy()
    bench = pd.DataFrame([{"benchmark": "the Fed holds", "right_%": 100 * (a == 0).mean(), "meetings": len(a)},
                          {"benchmark": "the Fed repeats its last decision", "right_%": 100 * (a[1:] == a[:-1]).mean(),
                           "meetings": len(a) - 1}])
    return per, pts, bench


def calibration(pts: pd.DataFrame, horizons=("1 week", "1 month")) -> pd.DataFrame:
    rows = []
    for h in horizons:
        d = pts[pts["horizon"] == h].copy()
        d["bucket"] = pd.cut(d["p"], BUCKETS, include_lowest=True)
        for b, g in d.groupby("bucket", observed=True):
            lo, hi = wilson(g["happened"].sum(), len(g))
            rows.append({"horizon": h, "bucket": str(b), "outcomes": len(g), "mean_forecast": g["p"].mean(),
                         "share_happened": g["happened"].mean(), "low": lo, "high": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B. Bond market
# ---------------------------------------------------------------------------
def daily_changes(km, pm, mt) -> pd.DataFrame:
    rates = pd.read_csv(config.FRED_DIR / "daily_rates.csv", parse_dates=["date"]).set_index("date")
    days = rates.dropna(subset=["DGS2"]).index
    days = days[days >= pd.Timestamp(config.START)]
    t = np.array([int(pd.Timestamp(f"{d.date()} 16:00", tz=config.TZ).timestamp()) for d in days])
    path = {}
    for _, m in mt.iterrows():
        for v in ("kalshi", "polymarket"):
            s = probs.hourly_series(v, m, km, pm)
            if s:
                path[(v, m["meeting"])] = probs.summarise(probs.on_grid(s, t, max_age=48 * 3600))["exp_bp"].to_numpy()
    decision = np.array([int(x.timestamp()) for x in mt["decision_time"]])
    rows = []
    for i in range(1, len(t)):
        use = []
        for j in [j for j in range(len(mt)) if decision[j] > t[i]][:2]:     # same meetings on both days
            vals = [path.get((v, mt["meeting"].iloc[j])) for v in ("kalshi", "polymarket")]
            if all(x is not None and np.isfinite(x[[i - 1, i]]).all() for x in vals):
                use.append(j)
            else:
                break
        if not use:
            continue
        rec = {"date": days[i], "meetings": len(use)}
        for v in ("kalshi", "polymarket"):
            rec[v] = sum(path[(v, mt["meeting"].iloc[j])][i] - path[(v, mt["meeting"].iloc[j])][i - 1] for j in use)
        for y in YIELDS:
            rec[y] = (rates[y].loc[days[i]] - rates[y].loc[days[i - 1]]) * 100
        rows.append(rec)
    return pd.DataFrame(rows).dropna()


def bond_tables(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for v in ("kalshi", "polymarket"):
        for y, name in YIELDS.items():
            fit = sm.OLS(d[v], sm.add_constant(d[y])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            n = len(d)
            lead = d[v].corr(d[y].shift(-1))           # prediction market today -> yield tomorrow
            lag = d[y].corr(d[v].shift(-1))            # yield today -> prediction market tomorrow
            rows.append({"venue": v, "yield": name, "days": n, "correlation": d[v].corr(d[y]),
                         "slope_bp_per_bp": fit.params[y], "t": fit.tvalues[y], "r2": fit.rsquared,
                         "corr_market_today_yield_tomorrow": lead, "corr_yield_today_market_tomorrow": lag,
                         "se_of_a_correlation": 1 / np.sqrt(n)})
    by_year = d.assign(year=d["date"].dt.year).groupby("year").apply(
        lambda g: pd.Series({"days": len(g), "kalshi_vs_6m_bill": g["kalshi"].corr(g["DTB6"]),
                             "polymarket_vs_6m_bill": g["polymarket"].corr(g["DTB6"]),
                             "kalshi_vs_polymarket": g["kalshi"].corr(g["polymarket"])}), include_groups=False)
    return pd.DataFrame(rows), by_year


def main() -> None:
    km, pm, mt = probs.markets()
    per, pts, bench = accuracy(km, pm, mt)
    cal = calibration(pts)
    per["horizon"] = pd.Categorical(per["horizon"], categories=list(HORIZONS), ordered=True)
    summary = (per.groupby(["horizon", "venue"], observed=True)
               .agg(meetings=("brier", "size"), favourite_right_pct=("favourite_right", lambda s: 100 * s.mean()),
                    median_p_of_what_happened=("p_of_what_happened", "median"), mean_brier=("brier", "mean"))
               .reset_index())
    per.round(4).to_csv(config.RESULTS / "07_forecasts_by_meeting.csv", index=False)
    summary.round(3).to_csv(config.RESULTS / "07_accuracy.csv", index=False)
    cal.round(3).to_csv(config.RESULTS / "07_calibration.csv", index=False)
    print(f"A. Accuracy over {per['meeting'].nunique()} decided meetings")
    for _, r in summary.iterrows():
        print(f"   {r['horizon']:8s} before  {LABEL[r['venue']]:10s} {r['meetings']:2d} meetings: favourite happened "
              f"{r['favourite_right_pct']:5.1f}%, median probability of what happened {r['median_p_of_what_happened']:.2f}, "
              f"Brier {r['mean_brier']:.3f}")
    for _, r in bench.iterrows():
        print(f"   benchmark '{r['benchmark']}': right {r['right_%']:.0f}% of {r['meetings']} meetings")
    miss = per[(per["horizon"] == "1 week") & ~per["favourite_right"]]
    for _, r in miss.iterrows():
        print(f"   a week before {r['meeting']}, {LABEL[r['venue']]} favoured {r['favourite_bp']:+.0f} bp "
              f"({100 * r['favourite_p']:.0f}%); the Fed did {r['actual_bp']:+.0f} bp")
    print("   calibration (both venues pooled): forecast -> share that happened [95% CI]")
    for _, r in cal.iterrows():
        print(f"     {r['horizon']:8s} {r['bucket']:16s} n={r['outcomes']:3d}  {r['mean_forecast']:.2f} -> "
              f"{r['share_happened']:.2f} [{r['low']:.2f}, {r['high']:.2f}]")

    d = daily_changes(km, pm, mt)
    table, by_year = bond_tables(d)
    table.round(3).to_csv(config.RESULTS / "07_bonds.csv", index=False)
    by_year.round(3).to_csv(config.RESULTS / "07_bonds_by_year.csv")
    print(f"\nB. Day-on-day changes ({len(d)} days, {d['date'].min():%Y-%m-%d} to {d['date'].max():%Y-%m-%d}; "
          f"one standard error of a correlation ~{1 / np.sqrt(len(d)):.2f})")
    for _, r in table.iterrows():
        print(f"   {LABEL[r['venue']]:10s} vs {r['yield']:12s} corr {r['correlation']:.2f}   slope {r['slope_bp_per_bp']:.2f} "
              f"(t {r['t']:.1f})   market today -> yield tomorrow {r['corr_market_today_yield_tomorrow']:+.2f}   "
              f"yield today -> market tomorrow {r['corr_yield_today_market_tomorrow']:+.2f}")
    print("   correlation with the 6-month bill by year:")
    for y, r in by_year.iterrows():
        print(f"     {y}: {int(r['days']):3d} days   Kalshi {r['kalshi_vs_6m_bill']:.2f}   Polymarket "
              f"{r['polymarket_vs_6m_bill']:.2f}   (Kalshi vs Polymarket {r['kalshi_vs_polymarket']:.2f})")
    figure(cal, d, by_year)


def figure(cal, d, by_year) -> None:
    set_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for h, marker, color in (("1 week", "o", VENUE["kalshi"]), ("1 month", "s", VENUE["polymarket"])):
        g = cal[cal["horizon"] == h]
        a1.errorbar(g["mean_forecast"], g["share_happened"],
                    yerr=[g["share_happened"] - g["low"], g["high"] - g["share_happened"]],
                    fmt=marker, color=color, capsize=3, label=f"{h} before", markersize=6)
    a1.plot([0, 1], [0, 1], color=INK_2, linestyle="--", linewidth=0.8)
    a1.set_xlim(0, 1), a1.set_ylim(-0.02, 1.02)
    a1.set_xlabel("forecast probability (both venues)")
    a1.set_ylabel("share of outcomes that happened (95% CI)")
    a1.set_title("Calibration: on the dashed line = honest probabilities")
    a1.legend(loc="upper left")

    a2.scatter(d["DTB6"], d["kalshi"], s=8, alpha=0.35, color=VENUE["kalshi"], edgecolors="none")
    fit = np.polyfit(d["DTB6"], d["kalshi"], 1)
    x = np.linspace(d["DTB6"].quantile(0.005), d["DTB6"].quantile(0.995), 50)
    a2.plot(x, np.polyval(fit, x), color=INK_2, linewidth=1)
    a2.set_xlim(x.min(), x.max())
    a2.set_ylim(d["kalshi"].quantile(0.003), d["kalshi"].quantile(0.997))
    a2.set_xlabel("6-month T-bill yield, day-on-day change (bp)")
    a2.set_ylabel("Kalshi expected rate change,\nday-on-day change (bp)")
    yrs = ", ".join(f"{y} {r['kalshi_vs_6m_bill']:.2f}" for y, r in by_year.iterrows())
    a2.set_title(f"Moving with the bond market (corr {d['kalshi'].corr(d['DTB6']):.2f})")
    a2.text(0.02, 0.97, f"correlation by year: {yrs}", transform=a2.transAxes, va="top", fontsize=8, color=INK_2)
    fig.tight_layout()
    save(fig, "07_accuracy_bonds.png")


if __name__ == "__main__":
    main()
