"""
Load Kalshi minute-level data, engineer contract-level features, and build
the cross-asset merged dataset used by all downstream model scripts.

Run this first. Outputs saved to data/:
    btc_contracts.csv, eth_contracts.csv, merged_contracts.csv

Contract-level features built per asset:
    opening_pup       — P(UP) at minute 0 (primary signal; market's initial bet)
    momentum_1min     — P(UP)[min1] - P(UP)[min0] (very-early drift)
    momentum_3min     — P(UP)[min3] - P(UP)[min0] (3-minute directional drift)
    mean_pup_5min     — mean P(UP) over first 5 minutes (smoothed opening signal)
    pup_vol_5min      — std of P(UP) over first 5 minutes (intracontract uncertainty)
    conviction_spread — |P(UP)[0] - 0.5| * 2, range [0,1] (how one-sided the open is)
    volume_log        — log1p(volume at minute 0) (market activity proxy)
    hour_of_open      — UTC hour of contract open (intraday seasonality)
    prev_open_pup     — previous contract's opening P(UP) (market memory / autocorrelation)

Cross-asset features (in merged dataset only):
    btc_eth_divergence — BTC opening P(UP) minus ETH opening P(UP)

EDA figures saved to figures/eda/:
    calibration_panel.png         — opening P(UP) histogram by outcome + calibration curve (appendix slide 19)
    probability_convergence.png   — mean P(UP) by minute-in-contract, split by outcome (main deck slide 7)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*non-interactive.*")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR     = ROOT / "data"
DATA_BTC_DIR = ROOT / "data" / "btc"
DATA_ETH_DIR = ROOT / "data" / "eth"
DATA_CLEAN   = ROOT / "data" / "cleaned"
FIGURES_DIR  = ROOT / "figures" / "eda"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "font.size":        34,
})


# ---------------------------------------------------------------------------
# Load and clean minute-level data
# ---------------------------------------------------------------------------

def load_minute_data(asset: str) -> pd.DataFrame:
    df = pd.read_csv(
        ROOT / "data" / asset.lower() / f"kalshi_{asset.lower()}_prices.csv",
        parse_dates=["Timestamp"],
    )
    df["minute_in_contract"] = df.groupby("Market Ticker").cumcount()

    # Quality flags — used downstream to filter noisy contracts
    prob_ok = (df["P(UP)"] + df["P(DOWN)"] - 1).abs() <= 0.01
    no_dup  = ~df.duplicated(subset=["Timestamp", "Market Ticker"])
    size_ok = df.groupby("Market Ticker")["Timestamp"].transform("count") >= 15
    df["is_clean"] = prob_ok & no_dup & size_ok
    return df


# ---------------------------------------------------------------------------
# Build contract-level features
# ---------------------------------------------------------------------------

def build_contract_features(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    records = []
    for ticker, g in df.groupby("Market Ticker"):
        g = g.sort_values("Timestamp").reset_index(drop=True)
        if len(g) < 4:
            continue

        pup = g["P(UP)"].values
        t0  = g["Timestamp"].iloc[0]

        outcomes = g["Actual Outcome"].dropna()
        actual   = outcomes.iloc[0] if len(outcomes) else None
        target   = 1 if actual == "UP" else (0 if actual == "DOWN" else np.nan)

        vol = g["Volume"].iloc[0]

        records.append({
            "ticker":            ticker,
            "asset":             asset,
            "contract_open":     t0,
            "hour_of_open":      t0.hour,
            "opening_pup":       pup[0],
            "momentum_1min":     pup[1] - pup[0] if len(pup) >= 2 else np.nan,
            "momentum_3min":     pup[3] - pup[0],
            "mean_pup_5min":     pup[:5].mean()         if len(pup) >= 5 else np.nan,
            "pup_vol_5min":      float(np.std(pup[:5])) if len(pup) >= 5 else np.nan,
            "conviction_spread": abs(pup[0] - 0.5) * 2,
            # log1p keeps zero-volume contracts intact (log(1+0) = 0)
            "volume_log":        float(np.log1p(vol)) if pd.notna(vol) else np.nan,
            "actual_outcome":    actual,
            "target":            target,
            "volume":            vol,
            "n_rows":            len(g),
            "is_clean":          bool(g["is_clean"].all()),
        })

    df_out = pd.DataFrame(records).sort_values("contract_open").reset_index(drop=True)

    # Previous same-asset contract's opening signal — tests market memory /
    # autocorrelation. Shift after sorting chronologically so lag is meaningful.
    df_out["prev_open_pup"] = df_out["opening_pup"].shift(1)
    return df_out


# ---------------------------------------------------------------------------
# Merge BTC and ETH into a matched cross-asset dataset
# ---------------------------------------------------------------------------

def merge_assets(btc_c: pd.DataFrame, eth_c: pd.DataFrame) -> pd.DataFrame:
    btc_m = btc_c.assign(open_key=btc_c["contract_open"].dt.floor("min"))
    eth_m = eth_c.assign(open_key=eth_c["contract_open"].dt.floor("min"))
    btc_m = btc_m.rename(columns={c: f"{c}_btc" for c in btc_m.columns if c != "open_key"})
    eth_m = eth_m.rename(columns={c: f"{c}_eth" for c in eth_m.columns if c != "open_key"})

    merged = (
        pd.merge(btc_m, eth_m, on="open_key")
        .sort_values("open_key")
        .reset_index(drop=True)
    )
    # Positive divergence = market more bullish on BTC than ETH at same moment
    merged["btc_eth_divergence"] = merged["opening_pup_btc"] - merged["opening_pup_eth"]
    return merged


# ---------------------------------------------------------------------------
# EDA: opening P(UP) distribution and calibration
# ---------------------------------------------------------------------------

def plot_pup_distribution(btc_c: pd.DataFrame) -> None:
    """
    Left panel: histogram of opening P(UP), split by actual outcome.
    Right panel: calibration curve — how well does opening P(UP) predict the outcome?
    A perfectly calibrated market lies on the diagonal (predicted = realized).
    """
    df = btc_c[btc_c["target"].notna()].copy()
    df["target"] = df["target"].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(20.8, 8.0))

    # Histogram split by outcome
    bins = np.linspace(0.30, 0.90, 28)
    up_mask = df["target"] == 1
    axes[0].hist(df.loc[up_mask,  "opening_pup"], bins=bins, alpha=0.65,
                 color="steelblue", density=True, label="Actual UP")
    axes[0].hist(df.loc[~up_mask, "opening_pup"], bins=bins, alpha=0.65,
                 color="tomato",    density=True, label="Actual DOWN")
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.7,
                    label="P(UP) = 0.5")
    axes[0].set_xlabel("Opening P(UP)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Opening P(UP) Distribution by Outcome  (BTC)", fontsize=28)
    axes[0].legend(loc="lower right", fontsize=20, framealpha=0.95)

    # Calibration: for each P(UP) bin, what fraction actually ended UP?
    cal_edges = np.linspace(0.30, 0.90, 13)
    centers, acc_vals, ci_vals = [], [], []
    for lo, hi in zip(cal_edges[:-1], cal_edges[1:]):
        mask = (df["opening_pup"] >= lo) & (df["opening_pup"] < hi)
        n = mask.sum()
        if n < 10:
            continue
        acc = df.loc[mask, "target"].mean()
        centers.append((lo + hi) / 2)
        acc_vals.append(acc)
        ci_vals.append(1.96 * np.sqrt(acc * (1 - acc) / n))

    axes[1].plot([0.3, 0.9], [0.3, 0.9], "k--", linewidth=1, alpha=0.5,
                 label="Perfect calibration")
    axes[1].errorbar(centers, acc_vals, yerr=ci_vals, fmt="o-", color="steelblue",
                     capsize=4, linewidth=1.8, markersize=6,
                     label="Realized accuracy ± 95% CI")
    axes[1].set_xlabel("Opening P(UP)")
    axes[1].set_ylabel("Fraction Actually UP")
    axes[1].set_title("Calibration: Opening P(UP) vs Realized Outcome  (BTC)", fontsize=28)
    axes[1].legend(loc="lower right", fontsize=20, framealpha=0.95)
    axes[1].set_xlim(0.28, 0.92)
    axes[1].set_ylim(0.28, 0.92)

    # Suptitle further from the panels so the panel titles read on their own.
    plt.suptitle("BTC Kalshi 15-min Contracts:  Opening Signal Analysis", y=0.93)
    fig.text(0.5, -0.02,
             "BTC opening P(UP) distribution by outcome (left) and calibration curve (right).",
             ha="center", va="top", fontsize=26, style="italic", color="#444")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_panel.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → make_figures/calibration_panel.png")


# ---------------------------------------------------------------------------
# EDA: P(UP) evolution within each contract
# ---------------------------------------------------------------------------

def plot_signal_evolution(btc_min: pd.DataFrame) -> None:
    """
    Mean P(UP) at each of the 15 minutes within a contract, split by outcome.
    If the market is informationally efficient, P(UP) should drift monotonically
    toward the eventual outcome as the expiry approaches.
    Shaded band = 95% CI across all contracts.
    """
    df = btc_min[btc_min["Actual Outcome"].notna()].copy()
    df = df[df["minute_in_contract"] < 15]
    df["ended_up"] = (df["Actual Outcome"] == "UP").astype(int)

    evo = (
        df.groupby(["minute_in_contract", "ended_up"])["P(UP)"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    evo["ci"] = 1.96 * evo["std"] / np.sqrt(evo["n"])

    fig, ax = plt.subplots(figsize=(16.0, 8.0))
    for outcome, label, color in [(1, "Contracts that ended UP",   "steelblue"),
                                   (0, "Contracts that ended DOWN", "tomato")]:
        sub = evo[evo["ended_up"] == outcome]
        ax.plot(sub["minute_in_contract"], sub["mean"], "-", color=color,
                linewidth=2.2, label=label)
        ax.fill_between(sub["minute_in_contract"],
                        sub["mean"] - sub["ci"],
                        sub["mean"] + sub["ci"],
                        alpha=0.15, color=color)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Minute in 15-min Contract Window")
    ax.set_ylabel("Mean P(UP)")
    ax.set_title("P(UP) Evolution Within Contract  (BTC)")
    ax.legend()
    ax.set_xticks(range(15))
    fig.text(0.5, -0.02,
             "Mean P(UP) trajectory by minute-in-contract, split by realized outcome.  "
             "Shaded band = 95% CI across 2,264 BTC contracts.",
             ha="center", va="top", fontsize=26, style="italic", color="#444")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "probability_convergence.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → make_figures/probability_convergence.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading minute-level data...")
    btc_min = load_minute_data("btc")
    eth_min = load_minute_data("eth")

    print("Building contract-level features...")
    btc_c = build_contract_features(btc_min, "BTC")
    eth_c = build_contract_features(eth_min, "ETH")
    merged = merge_assets(btc_c, eth_c)

    btc_c.to_csv(DATA_BTC_DIR / "btc_contracts.csv", index=False)
    eth_c.to_csv(DATA_ETH_DIR / "eth_contracts.csv", index=False)
    merged.to_csv(DATA_CLEAN / "merged_contracts.csv", index=False)

    valid = merged[merged["target_btc"].notna() & merged["target_eth"].notna()]
    print(f"\nBTC contracts  : {len(btc_c):,}  ({btc_c['target'].notna().sum():,} with outcomes)")
    print(f"ETH contracts  : {len(eth_c):,}  ({eth_c['target'].notna().sum():,} with outcomes)")
    print(f"Matched pairs  : {len(merged):,}  ({len(valid):,} both outcomes known)")
    print(f"Date range     : {merged['open_key'].min().date()} → {merged['open_key'].max().date()}")
    print("\nFeatures built per asset:")
    feat_cols = [c for c in btc_c.columns if c not in
                 ("ticker", "asset", "contract_open", "actual_outcome", "volume", "n_rows", "is_clean")]
    for f in feat_cols:
        print(f"  {f}")
    print("  btc_eth_divergence  (in merged only)")
    print("\nSaved → data/btc_contracts.csv, eth_contracts.csv, merged_contracts.csv")

    print("\n--- EDA Visualizations ---")
    plot_pup_distribution(btc_c)         # → figures/eda/calibration_panel.png
    plot_signal_evolution(btc_min)       # → figures/eda/probability_convergence.png
    # plot_market_regime + plot_cross_asset removed — out of scope for the
    # current presentation (no slide uses them).
    print("\nAll EDA figures saved to figures/eda/")


if __name__ == "__main__":
    main()
