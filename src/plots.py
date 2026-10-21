"""Shared chart style (same look as project 1)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt

from . import config

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
VENUE = {"kalshi": AQUA, "polymarket": BLUE}          # prediction markets
MARKET = {"ES": ORANGE, "USDJPY": YELLOW}              # fast benchmark markets
LABEL = {"kalshi": "Kalshi", "polymarket": "Polymarket", "ES": "S&P 500 futures", "USDJPY": "USD/JPY"}


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "figure.dpi": 110, "savefig.dpi": 160,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "lines.linewidth": 1.6, "legend.frameon": False,
    })


def save(fig, name: str) -> None:
    path = config.FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved figure -> figures/{name}")
