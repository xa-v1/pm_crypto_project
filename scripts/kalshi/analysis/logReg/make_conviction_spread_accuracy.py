"""
Market disagreement signal: does higher Kalshi opening conviction predict
higher realized accuracy on BTC 15-min contracts?

For each conviction-spread bin (|P(UP) − 0.5| × 2, range [0,1]), computes the
realized accuracy and 95% CI. This is appendix slide 20 of the presentation —
Market Disagreement Signal.

Reads : data/cleaned/merged_contracts.csv
Output: figures/eda/conviction_spread_accuracy.png
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family":      "Times New Roman",
    "font.serif":       ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size":        34,
    "axes.titlesize":   36,
    "axes.labelsize":   34,
    "xtick.labelsize":  32,
    "ytick.labelsize":  32,
    "legend.fontsize":  32,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.6,
})

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data" / "cleaned" / "merged_contracts.csv"
OUT  = ROOT / "figures" / "eda" / "conviction_spread_accuracy.png"
TARGET = "target_btc"


def main():
    df = pd.read_csv(DATA, parse_dates=["open_key"])
    df = df.dropna(subset=[TARGET, "conviction_spread_btc"]).copy()
    df[TARGET] = df[TARGET].astype(int)

    bins = np.arange(0.0, 1.01, 0.1)
    labels = [f"{b:.1f}–{b + 0.1:.1f}" for b in bins[:-1]]
    df["cs_bin"] = pd.cut(df["conviction_spread_btc"], bins=bins,
                          labels=labels, right=False)

    stats = (
        df.groupby("cs_bin", observed=True)[TARGET]
        .agg(accuracy="mean", n="count", std="std")
        .reset_index()
    )
    stats["ci95"] = 1.96 * stats["std"] / np.sqrt(stats["n"])

    fig, ax = plt.subplots(figsize=(22.0, 12.0))
    ax.bar(range(len(stats)), stats["accuracy"], yerr=stats["ci95"], capsize=6,
           color="steelblue", alpha=0.85, error_kw={"linewidth": 1.8},
           label="Realized accuracy")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, label="50% baseline")

    for i, row in stats.iterrows():
        ax.text(i, 0.30, f"n={int(row['n'])}", ha="center", fontsize=28,
                color="dimgray")

    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats["cs_bin"], rotation=40, ha="right")
    ax.set_xlabel("Conviction spread  |P(UP) − 0.5| × 2")
    ax.set_ylabel("Realized accuracy")
    ax.set_title(
        "Market disagreement signal:  BTC Kalshi 15-min contracts\n"
        "Does higher opening conviction predict higher realized accuracy?",
        pad=16,
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0.28, 1.05)
    ax.legend(loc="upper left", framealpha=0.95)

    fig.text(0.5, -0.02,
             "Realized accuracy by Kalshi opening conviction bin.  Error bars = 95% CI.  "
             "Bars above 0.5 imply directional predictive value at that conviction level.",
             ha="center", va="top", fontsize=22, style="italic", color="#444")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
