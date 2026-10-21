"""
STEP 5 - Economic significance: can you profit from the prediction markets' lag?

Step 4 found that after CPI and payrolls the prediction markets need 30-60
seconds to do half of their reaction, while USD/JPY needs about 3. The trade:

  1. Signal: USD/JPY's move in the first 5 seconds after 08:30:00. Up (dollar
     stronger) = hawkish news, rates expected higher; down = dovish. The rule only
     acts if the move is bigger than 3x USD/JPY's normal 5-second move in the
     5 minutes before (fixed in advance; in this sample it never binds).
  2. Contract: the FAVOURITE outcome of the next FOMC meeting (the most likely,
     and the most traded) on Kalshi and on Polymarket. If the news makes it more
     likely we buy YES; if less likely we sell YES (= buy NO). "More likely" = the
     favourite is the higher-rate outcome of the top two and the news is hawkish,
     or the lower-rate one and the news is dovish.
  3. Entry: decision at +6 s (5 s of signal + 1 s to react). We pay the first real
     trade on our side after that: the price at which another trader BOUGHT (if we
     buy) or SOLD (if we sell) - i.e. we pay the spread.
  4. Exit: 5 minutes after the release, at the first real trade on the other side.
  5. Costs: taker fees on entry and exit. Kalshi: 0.07 x p x (1-p) per contract
     (1.75 cents at p = 0.50). Polymarket: its own fee rate x p x (1-p) (0.05 on the
     2026 Fed markets; zero before).

P&L is in cents per contract (a contract pays $1). Robustness: later entries
(15, 30, 60 s), a 15-minute exit, one cent worse on each side, and the opposite
trade (should lose about as much as the real one makes).

Design note (honesty): a first version traded the meeting whose outcome was most
uncertain, often a later meeting; its contracts barely traded at 08:30 and most
orders could not be filled. The favourite of the NEXT meeting was chosen for
liquidity, before looking at profits; every variant tried is reported.

Run:  python 05_lag_trade.py        Writes results/05_*.csv and figures/05_lag_trade.png
"""
from __future__ import annotations

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, events, probs
from src.plots import INK_2, LABEL, VENUE, save, set_style

SIGNAL_S, REACT_S = 5, 1            # seconds of USD/JPY signal, seconds to act
EXIT_S = 300
NOISE_K = 3.0
FILL_WINDOW = {"entry": 60, "exit": 600}   # seconds to find a trade to fill against
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def unix(t) -> np.ndarray:
    return ((pd.to_datetime(t, utc=True) - EPOCH) / pd.Timedelta(seconds=1)).to_numpy()


def at(s: pd.Series, t: float) -> float:
    i = np.searchsorted(s.index.to_numpy(), t, side="right") - 1
    return s.iloc[i] if i >= 0 else np.nan


def kalshi_fee(p: float) -> float:
    return 0.07 * p * (1 - p)


def prints(venue, m, bp, km, pm, kt, pt):
    """Trades in one outcome as YES prices: (prints at the ask = someone bought YES,
    prints at the bid = someone sold YES, fee function)."""
    if venue == "kalshi":
        c = km[(km["event"] == m["kalshi_event"]) & (km["bp"] == bp)]
        if not len(c) or not len(kt):
            return None
        x = kt[kt["ticker"] == c["ticker"].iloc[0]]
        buy = (x["taker_side"] == "yes").to_numpy()
        ts, p, q = x["ts"].to_numpy(), x["yes_price"].to_numpy(), x["count"].to_numpy()
        fee = kalshi_fee
    else:
        c = pm[(pm["event"] == m["poly_event"]) & (pm["bp"] == bp)]
        if not len(c) or not len(pt):
            return None
        x = pt[pt["conditionId"] == c["condition_id"].iloc[0]]
        yes = (x["outcome"] == "Yes").to_numpy()
        buy = (yes & (x["side"] == "BUY").to_numpy()) | (~yes & (x["side"] == "SELL").to_numpy())
        ts, q = x["timestamp"].to_numpy(), x["size"].to_numpy()
        p = np.where(yes, x["price"].to_numpy(), 1 - x["price"].to_numpy())
        rate = c["fee_rate"].iloc[0]
        fee = (lambda pr, rate=rate: rate * pr * (1 - pr))
    ask = pd.DataFrame({"ts": ts[buy], "p": p[buy], "q": q[buy]})
    bid = pd.DataFrame({"ts": ts[~buy], "p": p[~buy], "q": q[~buy]})
    return ask, bid, fee


def first_print(df: pd.DataFrame, a: float, b: float):
    d = df[(df["ts"] >= a) & (df["ts"] < b)].sort_values("ts", kind="stable")
    return d.iloc[0] if len(d) else None


def backtest(km, pm, mt, rel, entry_s=SIGNAL_S + REACT_S, exit_s=EXIT_S, opposite=False,
             worse_c=0.0) -> pd.DataFrame:
    rows = []
    for _, r in rel.iterrows():
        t0 = r["time"].tz_convert("UTC")
        T = int(t0.timestamp())
        tag = f"{r['kind']}_{r['time']:%Y-%m-%d}"
        f = config.DATA_LSE_1S / f"USDJPY_{tag}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        if len(d) < 100:
            continue
        fx = pd.Series(np.log(d["close"].to_numpy()) * 1e4, index=unix(d["timestamp"]).astype("int64") + 1)
        signal = at(fx, T + SIGNAL_S) - at(fx, T)
        noise = np.nanstd([at(fx, t + SIGNAL_S) - at(fx, t) for t in range(T - 300, T - 10, SIGNAL_S)])
        if not np.isfinite(signal) or abs(signal) < NOISE_K * noise:
            continue
        m = probs.next_meeting(mt, t0)
        s = probs.minute_series("polymarket", tag, m, km, pm, start_ts=T - 3 * 3600)
        p0 = probs.on_grid(s, np.array([T - 60]), max_age=3 * 86400).iloc[0].dropna().sort_values(ascending=False)
        if len(p0) < 2:
            continue
        fav, second = p0.index[0], p0.index[1]
        direction = 1 if (signal > 0) == (fav > second) else -1      # +1 buy YES, -1 sell YES
        if opposite:
            direction = -direction
        kt = pd.read_parquet(config.KALSHI_DIR / "trades" / f"{tag}.parquet")
        pt = pd.read_parquet(config.POLY_DIR / "trades" / f"{tag}.parquet")
        if len(kt):
            kt["ts"] = unix(kt["time"])
        for venue in ("kalshi", "polymarket"):
            row = {"release": tag, "kind": r["kind"], "year": r["time"].year, "venue": venue,
                   "signal_bp": signal, "signal_vs_noise": abs(signal) / noise, "meeting": m["meeting"],
                   "favourite_bp": fav, "favourite_p": p0.iloc[0], "direction": direction, "filled": False}
            book = prints(venue, m, fav, km, pm, kt, pt)
            if book is not None:
                ask, bid, fee = book
                inside, outside = (ask, bid) if direction > 0 else (bid, ask)
                e = first_print(inside, T + entry_s, T + entry_s + FILL_WINDOW["entry"])
                x = first_print(outside, T + exit_s, T + exit_s + FILL_WINDOW["exit"])
                if e is not None and x is not None:
                    gross = direction * (x["p"] - e["p"]) - 2 * worse_c / 100
                    near = inside[(inside["ts"] >= e["ts"]) & (inside["ts"] < e["ts"] + 30)
                                  & (direction * (inside["p"] - e["p"]) <= 0.02)]
                    row.update(filled=True, entry_price=e["p"], exit_price=x["p"], entry_after_s=e["ts"] - T,
                               gross_c=100 * gross, fees_c=100 * (fee(e["p"]) + fee(x["p"])),
                               capacity_contracts=near["q"].sum(), capacity_usd=(near["q"] * near["p"]).sum())
                    row["net_c"] = row["gross_c"] - row["fees_c"]
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(label: str, bt: pd.DataFrame) -> list[dict]:
    rng = np.random.default_rng(5)
    out = []
    for v in ("kalshi", "polymarket"):
        g = bt[(bt["venue"] == v) & bt["filled"]]
        x = g["net_c"].to_numpy()
        boot = np.array([rng.choice(x, len(x)).mean() for _ in range(5000)]) if len(x) > 1 else np.array([np.nan])
        out.append({"variant": label, "venue": v, "signals": int((bt["venue"] == v).sum()), "filled": len(g),
                    "gross_c": g["gross_c"].mean(), "fees_c": g["fees_c"].mean(), "net_c": x.mean() if len(x) else np.nan,
                    "net_low": np.percentile(boot, 2.5), "net_high": np.percentile(boot, 97.5),
                    "win_rate": (x > 0).mean() if len(x) else np.nan,
                    "entry_after_s": g["entry_after_s"].median() if len(g) else np.nan,
                    "capacity_usd_median": g["capacity_usd"].median() if len(g) else np.nan})
    return out


def main() -> None:
    km, pm, mt = probs.markets()
    pm["fee_rate"] = pm["fee_schedule"].map(lambda s: json.loads(s).get("rate", 0.0) if isinstance(s, str) else 0.0)
    rel = events.load_events({k: events.RELEASES[k] for k in ("CPI", "Payrolls")})
    rel = rel[rel["time"] >= pd.Timestamp(config.SECONDS_FROM, tz=config.TZ)].reset_index(drop=True)

    main_bt = backtest(km, pm, mt, rel)
    main_bt.to_csv(config.RESULTS / "05_trades.csv", index=False)
    rows = summarise("main: act at 6 s, exit at 5 min", main_bt)
    for y in (2025, 2026):
        rows += summarise(f"main, {y} only", main_bt[main_bt["year"] == y])
    close = main_bt["favourite_p"] < 0.85            # a close call: the favourite is below 85%
    rows += summarise("main, favourite below 85%", main_bt[close])
    rows += summarise("main, favourite 85% or more", main_bt[~close])
    delays = {}
    for s in (15, 30, 60):
        delays[s] = backtest(km, pm, mt, rel, entry_s=s)
        rows += summarise(f"act at {s} s", delays[s])
    rows += summarise("exit at 15 min", backtest(km, pm, mt, rel, exit_s=900))
    rows += summarise("1 cent worse on entry and exit", backtest(km, pm, mt, rel, worse_c=1.0))
    rows += summarise("opposite trade (sanity check)", backtest(km, pm, mt, rel, opposite=True))
    out = pd.DataFrame(rows)
    out.round(3).to_csv(config.RESULTS / "05_lag_trade_summary.csv", index=False)

    n_sig = main_bt["release"].nunique()
    print(f"{len(rel)} CPI and payrolls releases since {config.SECONDS_FROM}; USD/JPY gave a signal at {n_sig} "
          f"(its 5-second move was at least {main_bt['signal_vs_noise'].min():.0f}x normal every time)")
    filled = main_bt[main_bt["filled"]]
    print("By year: median USD/JPY 5-second move (bp), median favourite probability, median seconds to our fill")
    for y, g in filled.groupby("year"):
        print(f"   {y}: {g['signal_bp'].abs().median():5.1f} bp   favourite {g['favourite_p'].median():.2f}   "
              f"fill after {g['entry_after_s'].median():4.0f} s")
    print("\nNet profit per contract after spread and fees, cents (95% CI); win rate; how much traded on our side")
    for _, r in out.iterrows():
        print(f"   {r['variant']:34s} {LABEL[r['venue']]:10s} {r['filled']:2d} trades   gross {r['gross_c']:5.2f}   "
              f"fees {r['fees_c']:4.2f}   net {r['net_c']:5.2f} [{r['net_low']:5.2f}, {r['net_high']:5.2f}]   "
              f"win {100 * r['win_rate']:3.0f}%   fill after {r['entry_after_s']:4.0f} s   "
              f"~${r['capacity_usd_median']:,.0f}")
    figure(main_bt, delays, out)


def figure(main_bt, delays, out) -> None:
    set_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={"width_ratios": [1, 1.3]})
    labels = {6: "main: act at 6 s, exit at 5 min", 15: "act at 15 s", 30: "act at 30 s", 60: "act at 60 s"}
    for j, v in enumerate(("kalshi", "polymarket")):
        xs, ys, lo, hi = [], [], [], []
        for s, lab in labels.items():
            r = out[(out["variant"] == lab) & (out["venue"] == v)].iloc[0]
            xs.append(s + (j - 0.5) * 1.5)
            ys.append(r["net_c"])
            lo.append(r["net_c"] - r["net_low"])
            hi.append(r["net_high"] - r["net_c"])
        a1.errorbar(xs, ys, yerr=[lo, hi], color=VENUE[v], marker="o", capsize=3, label=LABEL[v], linewidth=1.6)
    a1.axhline(0, color=INK_2, linewidth=0.8)
    a1.set_xticks(list(labels), [f"{s} s" for s in labels])
    a1.set_xlabel("when you act after 08:30:00")
    a1.set_ylabel("net profit per contract (cents, 95% CI)")
    a1.set_title("The edge is gone after about 30 seconds")
    a1.legend()

    f = main_bt[main_bt["filled"]].copy()
    f["date"] = pd.to_datetime(f["release"].str[-10:])
    for v in ("kalshi", "polymarket"):
        g = f[f["venue"] == v]
        a2.scatter(g["date"], g["net_c"], color=VENUE[v], s=28, alpha=0.8, label=LABEL[v], edgecolors="none")
    a2.axhline(0, color=INK_2, linewidth=0.8)
    a2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    a2.set_ylabel("net profit per contract (cents)")
    a2.set_title("Every release, acting at 6 s (after spread and fees)")
    a2.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "05_lag_trade.png")


if __name__ == "__main__":
    main()
