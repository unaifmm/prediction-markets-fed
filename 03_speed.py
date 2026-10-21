"""
STEP 3 - Research question 1: how fast do prediction markets price macro news?

For every CPI and payrolls release since May 2023 we follow four markets minute by
minute, from 30 minutes before to 30 minutes after 08:30 New York:
  * Kalshi and Polymarket: the expected rate change over the next two FOMC
    decisions (bp; the next one only, if either venue does not list the second yet);
  * S&P 500 futures and USD/JPY: log price (bp).

Speed = the share of each market's 30-minute reaction already done h minutes
after the release (the "absorption curve"):

    S(h) = sum_i sign(R_i) * (X_i(t0 + h) - X_i(t0))  /  sum_i |R_i|,   R_i = X_i(t0 + 30) - X_i(t0)

Pooling the releases this way weights big reactions more than tiny ones, so a
release that barely moved a market does not add noise. S(1) = share done in the
release minute. 95% confidence intervals: bootstrap over releases (2,000 draws).
Only releases where all four markets have data are used, so the four curves
compare the same events.

Robustness: a 60-minute anchor; only the half of releases with the biggest news
(USD/JPY's 2-minute move); 2023-24 vs 2025-26; Polymarket's own time stamps.

Run:  python 03_speed.py        Writes results/03_*.csv and figures/03_speed.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, events, probs
from src.lse import load_lse
from src.plots import GRID, INK_2, LABEL, MARKET, VENUE, save, set_style

SERIES = ("kalshi", "polymarket", "ES", "USDJPY")
H = np.arange(-30, 61)                 # minutes around the release
K0 = int(np.where(H == 0)[0][0])
ANCHOR = 30                            # the reaction is measured over 30 minutes
N_BOOT = 2000


def expected_paths(tag, t0, mt, km, pm, grid) -> dict[str, np.ndarray] | None:
    """Expected rate change summed over the next two decisions, on both venues.
    A meeting counts only if BOTH venues have a complete series for it, so the two
    venues are always measured on the same meetings (the second one is dropped if
    either venue does not list it yet). Also returns Polymarket on its own time stamps."""
    start = int((t0 - config.WINDOW_BEFORE).timestamp())
    keys = ("kalshi", "polymarket", "polymarket_raw_stamps")
    total, n = {k: np.zeros(len(grid)) for k in keys}, 0
    for _, m in mt[mt["decision_time"] > t0 + pd.Timedelta(minutes=1)].head(2).iterrows():
        e = {}
        for k in keys:
            venue = "kalshi" if k == "kalshi" else "polymarket"
            s = probs.minute_series(venue, tag, m, km, pm, start_ts=start, align_polymarket=k != "polymarket_raw_stamps")
            e[k] = probs.summarise(probs.on_grid(s, grid, max_age=3 * 86400))["exp_bp"].to_numpy() if s else None
        if any(v is None or not np.isfinite(v).all() for v in e.values()):
            break
        total, n = {k: total[k] + e[k] for k in keys}, n + 1
    return total if n else None


def build_paths() -> tuple[pd.DataFrame, dict]:
    km, pm, mt = probs.markets()
    rel = events.load_events({k: events.RELEASES[k] for k in ("CPI", "Payrolls")})
    rel = rel[rel["time"] >= pd.Timestamp(config.START, tz=config.TZ)].reset_index(drop=True)
    px = np.log(load_lse(["ES", "USDJPY"], how="outer")) * 1e4
    info, paths = [], {s: [] for s in SERIES + ("polymarket_raw_stamps",)}
    for _, r in rel.iterrows():
        t0 = r["time"].tz_convert("UTC")
        tag = f"{r['kind']}_{r['time']:%Y-%m-%d}"
        grid = int(t0.timestamp()) + H * 60
        e = expected_paths(tag, t0, mt, km, pm, grid)
        for k in ("kalshi", "polymarket", "polymarket_raw_stamps"):
            paths[k].append(e[k] if e is not None else np.full(len(grid), np.nan))
        for name in ("ES", "USDJPY"):
            # price at minute boundary t = close of the 1-minute bar that starts at t - 1
            paths[name].append(px[name].reindex(r["time"] + pd.to_timedelta(H - 1, unit="min")).ffill().to_numpy())
        info.append({"release": tag, "kind": r["kind"], "year": r["time"].year, "surprise_z": r["surprise_z"]})
    return pd.DataFrame(info), {k: np.array(v) for k, v in paths.items()}


def curve(x: np.ndarray, anchor: int = ANCHOR) -> np.ndarray:
    r = x[:, K0 + anchor] - x[:, K0]
    return (np.sign(r)[:, None] * (x - x[:, [K0]])).sum(axis=0) / np.abs(r).sum()


def bootstrap(paths, idx, anchor=ANCHOR, series=SERIES) -> dict:
    rng = np.random.default_rng(7)
    draws = [rng.choice(idx, len(idx)) for _ in range(N_BOOT)]
    return {s: np.array([curve(paths[s][d], anchor) for d in draws]) for s in series}


def summary_rows(label, paths, idx, anchor=ANCHOR, series=SERIES) -> list[dict]:
    boot = bootstrap(paths, idx, anchor, series)
    rows = []
    for s in series:
        c = curve(paths[s][idx], anchor)
        lo, hi = np.percentile(boot[s], [2.5, 97.5], axis=0)
        h50 = next((h for h in range(0, anchor + 1) if c[K0 + h] >= 0.5), np.nan)
        h80 = next((h for h in range(0, anchor + 1) if c[K0 + h] >= 0.8), np.nan)
        rows.append({"sample": label, "series": s, "releases": len(idx),
                     "S1": c[K0 + 1], "S1_low": lo[K0 + 1], "S1_high": hi[K0 + 1],
                     "S2": c[K0 + 2], "S5": c[K0 + 5], "S5_low": lo[K0 + 5], "S5_high": hi[K0 + 5],
                     "S10": c[K0 + 10], "S15": c[K0 + 15], "minutes_to_50%": h50, "minutes_to_80%": h80})
    for s in ("kalshi", "polymarket"):
        if s in boot and "USDJPY" in boot:
            d = boot[s][:, K0 + 1] - boot["USDJPY"][:, K0 + 1]
            rows.append({"sample": label, "series": f"{s} minus USDJPY", "releases": len(idx),
                         "S1": d.mean(), "S1_low": np.percentile(d, 2.5), "S1_high": np.percentile(d, 97.5)})
    return rows


def main() -> None:
    info, paths = build_paths()
    ok = np.all([np.isfinite(paths[s]).all(axis=1) for s in SERIES], axis=0)
    idx = np.where(ok)[0]
    print(f"{len(info)} CPI and payrolls releases since {config.START}; all four markets available for {len(idx)}")
    news = np.abs(paths["USDJPY"][:, K0 + 2] - paths["USDJPY"][:, K0])
    big = np.where(ok & (news >= np.nanmedian(news[ok])))[0]
    late = np.where(ok & (info["year"] >= 2025).to_numpy())[0]
    early = np.where(ok & (info["year"] < 2025).to_numpy())[0]
    raw_ok = np.where(ok & np.isfinite(paths["polymarket_raw_stamps"]).all(axis=1))[0]

    rows = summary_rows("main (30-minute reaction)", paths, idx)
    rows += summary_rows("60-minute reaction", paths, idx, anchor=60)
    rows += summary_rows("biggest-news half", paths, big)
    rows += summary_rows("2023-24", paths, early)
    rows += summary_rows("2025-26", paths, late)
    rows += [r | {"series": "polymarket, own time stamps"} for r in
             summary_rows("main (30-minute reaction)", paths, raw_ok, series=("polymarket_raw_stamps",))]
    out = pd.DataFrame(rows)
    out.round(3).to_csv(config.RESULTS / "03_speed_summary.csv", index=False)

    print("\nShare of the 30-minute reaction done after h minutes (95% CI), all four markets on the same releases")
    for _, r in out[out["sample"] == "main (30-minute reaction)"].iterrows():
        if "minus" in r["series"]:
            print(f"   {r['series']:28s} first-minute share difference {r['S1']:+.2f} [{r['S1_low']:+.2f}, {r['S1_high']:+.2f}]")
        else:
            name = LABEL.get(r["series"], r["series"])
            print(f"   {name:28s} 1 min {r['S1']:.2f} [{r['S1_low']:.2f}, {r['S1_high']:.2f}]   2 min {r['S2']:.2f}   "
                  f"5 min {r['S5']:.2f}   10 min {r['S10']:.2f}   15 min {r['S15']:.2f}")
    print("\nFirst-minute share in the robustness samples:")
    for label in ("60-minute reaction", "biggest-news half", "2023-24", "2025-26"):
        d = out[(out["sample"] == label) & ~out["series"].str.contains("minus")].set_index("series")["S1"]
        n = out[out["sample"] == label]["releases"].iloc[0]
        print(f"   {label:20s} (n={n:2d})  " + "   ".join(f"{LABEL[s]} {d[s]:.2f}" for s in SERIES))

    curves = pd.DataFrame({LABEL[s]: curve(paths[s][idx]) for s in SERIES}, index=H)
    curves.index.name = "minutes_after_release"
    curves.loc[-5:30].round(4).to_csv(config.RESULTS / "03_speed_curves.csv")
    figure(paths, idx, early, late)


def figure(paths, idx, early, late) -> None:
    set_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    boot = bootstrap(paths, idx)
    hh = np.arange(-5, ANCHOR + 1)
    for s in SERIES:
        c = curve(paths[s][idx])[K0 - 5:K0 + ANCHOR + 1]
        color = VENUE.get(s, MARKET.get(s))
        a1.plot(hh, c, color=color, label=LABEL[s], linewidth=2 if s in VENUE else 1.3,
                linestyle="-" if s in VENUE else ":")
        if s in VENUE:
            lo, hi = np.percentile(boot[s], [2.5, 97.5], axis=0)
            a1.fill_between(hh, lo[K0 - 5:K0 + ANCHOR + 1], hi[K0 - 5:K0 + ANCHOR + 1], color=color, alpha=0.12, linewidth=0)
    a1.axvline(0, color=GRID, linewidth=6, zorder=0)
    a1.axhline(1, color=INK_2, linewidth=0.7, linestyle="--")
    a1.set_xlabel("minutes after the release (08:30 New York)")
    a1.set_ylabel("share of the 30-minute reaction done")
    a1.set_title(f"How fast the news is priced ({len(idx)} CPI and payrolls releases)")
    a1.legend(loc="lower right")

    width = 0.2
    for j, s in enumerate(SERIES):
        vals, errs = [], []
        for sample in (early, late):
            b = bootstrap(paths, sample, series=(s,))[s][:, K0 + 1]
            v = curve(paths[s][sample])[K0 + 1]
            vals.append(v)
            errs.append([v - np.percentile(b, 2.5), np.percentile(b, 97.5) - v])
        x = np.arange(2) + (j - 1.5) * width
        a2.bar(x, vals, width=width, color=VENUE.get(s, MARKET.get(s)), label=LABEL[s],
               yerr=np.array(errs).T, capsize=2, error_kw={"elinewidth": 0.8, "ecolor": INK_2})
    a2.set_xticks([0, 1], [f"2023-24 (n={len(early)})", f"2025-26 (n={len(late)})"])
    a2.set_ylabel("share done in the release minute")
    a2.set_title("First-minute share by period (95% CI)")
    a2.set_ylim(0, 1.15)
    a2.legend(ncol=2, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "03_speed.png")


if __name__ == "__main__":
    main()
