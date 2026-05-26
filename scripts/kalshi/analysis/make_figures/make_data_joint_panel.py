"""
Generate a single combined data figure intended for Xavi's data slide. Shows
all the data we have on one shared time axis. Saves to
make_figures/data_joint_panel.png.

Three stacked panels with a shared x-axis:
  1) Coinbase BTC 1-minute close
  2) Kalshi BTC opening P(UP) per contract (scatter, with 0.5 reference)
  3) Daily contracts per day (Kalshi coverage)

The 13-day scraper-outage gap is shaded in pink across all three panels so
it's visually unmistakable.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib as _mpl
_mpl.rcParams.update({
    "font.family":      "Times New Roman",
    "font.serif":       ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size":        34,
    "axes.titlesize":   36,
    "axes.labelsize":   34,
    "xtick.labelsize":  32,
    "ytick.labelsize":  32,
    "legend.fontsize":  32,
})


ROOT = Path(__file__).resolve().parents[4]
SPOT_CSV   = ROOT / "data" / "btc" / "spot_btc_1m.csv"
KALSHI_CSV = ROOT / "data" / "btc" / "kalshi_btc_prices.csv"
OUT_PNG    = ROOT / "figures" / "eda" / "data_joint_panel.png"

SPOT_C  = "#1f4e8c"
PUP_C   = "#c1432e"
CNT_C   = "#9aa8bd"
GAP_C   = "#f4d4d4"


def main():
    # Spot
    spot = pd.read_csv(SPOT_CSV, parse_dates=["open_time_utc"])
    spot = spot.sort_values("open_time_utc")

    # Kalshi
    df = pd.read_csv(KALSHI_CSV, parse_dates=["Timestamp"])
    df = df.sort_values(["Market Ticker", "Timestamp"])
    contracts = df.groupby("Market Ticker").agg(
        open_ts=("Timestamp", "min"),
        opening_pup=("P(UP)", "first"),
    ).reset_index()
    contracts["date"] = contracts["open_ts"].dt.floor("D")
    daily = contracts.groupby("date").size().reset_index(name="n_contracts")

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, sharex=True,
        figsize=(22.0, 14.0),
        gridspec_kw={"height_ratios": [3, 3, 1.2], "hspace": 0.10},
    )

    gap_start = pd.Timestamp("2026-02-26")
    gap_end   = pd.Timestamp("2026-03-11")
    for ax in (ax1, ax2, ax3):
        ax.axvspan(gap_start, gap_end, color=GAP_C, alpha=0.55, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel 1: Spot
    ax1.plot(spot["open_time_utc"], spot["close"],
             color=SPOT_C, linewidth=1.0, zorder=2)
    ax1.set_ylabel("Coinbase BTC-USD\nclose (USD)", fontsize=34)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax1.set_title(
        "Joint data panel: Kalshi PM and Coinbase spot, 15 Feb to 28 Mar 2026",
        fontsize=34, loc="left", pad=10,
    )
    ax1.grid(alpha=0.25)

    # Panel 2: Opening P(UP) scatter
    ax2.scatter(contracts["open_ts"], contracts["opening_pup"],
                s=5, color=PUP_C, alpha=0.30, edgecolor="none", zorder=2)
    ax2.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6,
                zorder=1)
    ax2.set_ylabel("Kalshi BTC\nopening P(UP)", fontsize=34)
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.25)

    # Panel 3: Coverage (contracts / day)
    ax3.bar(daily["date"], daily["n_contracts"], color=CNT_C, width=0.9,
            zorder=2)
    ax3.set_ylabel("Contracts\nper day", fontsize=34)
    ax3.set_xlabel("Date (UTC)", fontsize=34)
    ax3.set_ylim(0, 110)
    ax3.grid(alpha=0.25, axis="y")

    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax3.get_xticklabels(), fontsize=34)

    # Annotate the gap on the top panel
    ax1.annotate(
        "13-day scraper outage",
        xy=(gap_start + (gap_end - gap_start) / 2, ax1.get_ylim()[1] * 0.97),
        ha="center", va="top", fontsize=34, color="#995555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#995555", alpha=0.85),
    )

    fig.text(0.5, -0.01,
             "Coinbase BTC-USD close, Kalshi BTC opening P(UP), and contracts per day.  "
             "Pink shading marks a 13-day scraper outage.",
             ha="center", va="top", fontsize=22, style="italic", color="#444")

    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {OUT_PNG}")


if __name__ == "__main__":
    main()
