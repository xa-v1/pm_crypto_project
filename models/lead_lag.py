"""
BTC/ETH lead-lag analysis: cross-correlogram at lags -3 to +3 minutes
across matched 15-minute contract windows.

Positive lag k → BTC leads ETH (BTC at minute t correlated with ETH at t+k).
Negative lag k → ETH leads BTC.

Loads: data/kalshi_btc_prices.csv, data/kalshi_eth_prices.csv,
       data/merged_contracts.csv
Outputs:
    figures/cross_correlogram.png
    data/lead_lag_result.json   (peak_lag, peak_corr, p_value, significant)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

MAX_LAG = 3


# ---------------------------------------------------------------------------
# Load and align minute-level data within matched contract windows
# ---------------------------------------------------------------------------

def load_aligned_windows() -> list[tuple[np.ndarray, np.ndarray]]:
    btc = pd.read_csv(DATA_DIR / "kalshi_btc_prices.csv", parse_dates=["Timestamp"])
    eth = pd.read_csv(DATA_DIR / "kalshi_eth_prices.csv", parse_dates=["Timestamp"])
    merged = pd.read_csv(DATA_DIR / "merged_contracts.csv", parse_dates=["open_key"])

    btc["minute_in_contract"] = btc.groupby("Market Ticker").cumcount()
    eth["minute_in_contract"] = eth.groupby("Market Ticker").cumcount()

    # Index by (ticker → sorted P(UP) series)
    btc_series = {
        ticker: g.sort_values("minute_in_contract")["P(UP)"].values
        for ticker, g in btc.groupby("Market Ticker")
    }
    eth_series = {
        ticker: g.sort_values("minute_in_contract")["P(UP)"].values
        for ticker, g in eth.groupby("Market Ticker")
    }

    windows = []
    for _, row in merged.iterrows():
        b_pup = btc_series.get(row["ticker_btc"])
        e_pup = eth_series.get(row["ticker_eth"])
        if b_pup is None or e_pup is None:
            continue
        n = min(len(b_pup), len(e_pup))
        if n < MAX_LAG + 3:
            continue
        windows.append((b_pup[:n], e_pup[:n]))

    return windows


# ---------------------------------------------------------------------------
# Per-window cross-correlation at each lag
# ---------------------------------------------------------------------------

def window_xcorr(btc_pup: np.ndarray, eth_pup: np.ndarray, max_lag: int) -> dict[int, float]:
    n = len(btc_pup)
    result = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = btc_pup[: n - lag], eth_pup[lag:]
        else:
            a, b = btc_pup[-lag:], eth_pup[: n + lag]
        if len(a) < 3:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if not np.isnan(r):
            result[lag] = r
    return result


# ---------------------------------------------------------------------------
# Aggregate and plot
# ---------------------------------------------------------------------------

def compute_correlogram(windows: list) -> dict:
    from collections import defaultdict
    lag_samples: dict[int, list] = defaultdict(list)

    for btc_pup, eth_pup in windows:
        xcorr = window_xcorr(btc_pup, eth_pup, MAX_LAG)
        for lag, r in xcorr.items():
            lag_samples[lag].append(r)

    lags = sorted(lag_samples.keys())
    means = np.array([np.mean(lag_samples[k]) for k in lags])
    ses = np.array([np.std(lag_samples[k]) / np.sqrt(len(lag_samples[k])) for k in lags])
    ns = np.array([len(lag_samples[k]) for k in lags])

    # t-test vs zero for each lag
    t_stats = means / ses
    p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

    peak_idx = np.argmax(np.abs(means))
    peak_lag = lags[peak_idx]
    peak_corr = means[peak_idx]
    peak_p = p_values[peak_idx]

    return {
        "lags": lags,
        "means": means,
        "ses": ses,
        "ns": ns,
        "p_values": p_values,
        "peak_lag": peak_lag,
        "peak_corr": float(peak_corr),
        "peak_p": float(peak_p),
    }


def plot_correlogram(corr: dict) -> None:
    lags = corr["lags"]
    means = corr["means"]
    ci = 1.96 * corr["ses"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(lags, means, color="steelblue", alpha=0.75, width=0.6, label="Mean cross-corr")
    ax.errorbar(lags, means, yerr=ci, fmt="none", color="black", capsize=5, linewidth=1.2,
                label="95% CI")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    # Mark peak lag
    pk = corr["peak_lag"]
    ax.bar([pk], [means[lags.index(pk)]], color="tomato", alpha=0.9, width=0.6,
           label=f"Peak lag = {pk}  (r={corr['peak_corr']:+.3f}, p={corr['peak_p']:.3f})")

    # Significance stars
    for i, lag in enumerate(lags):
        p = corr["p_values"][i]
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        if star:
            ax.text(lag, means[i] + np.sign(means[i]) * (ci[i] + 0.005),
                    star, ha="center", va="bottom", fontsize=9, color="black")

    ax.set_xlabel("Lag (minutes)  [positive = BTC leads ETH]")
    ax.set_ylabel("Mean Pearson r  (averaged across contract windows)")
    ax.set_title(
        f"BTC / ETH P(UP) Cross-Correlogram  "
        f"(n = {corr['ns'][len(lags)//2]:,} windows per lag)\n"
        f"* p<0.05   ** p<0.01   *** p<0.001"
    )
    ax.set_xticks(lags)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_correlogram.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → figures/cross_correlogram.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading and aligning matched contract windows...")
    windows = load_aligned_windows()
    print(f"  Matched windows: {len(windows):,}")

    print("Computing cross-correlogram...")
    corr = compute_correlogram(windows)

    print("\nCross-correlogram results:")
    print(f"{'Lag':>6}  {'Mean r':>8}  {'SE':>8}  {'p-value':>10}  {'n':>6}")
    print("-" * 45)
    for i, lag in enumerate(corr["lags"]):
        sig = " *" if corr["p_values"][i] < 0.05 else ""
        print(f"{lag:>6}  {corr['means'][i]:>8.4f}  {corr['ses'][i]:>8.4f}  "
              f"{corr['p_values'][i]:>10.4f}  {corr['ns'][i]:>6}{sig}")

    pk = corr["peak_lag"]
    direction = "BTC leads ETH" if pk > 0 else ("ETH leads BTC" if pk < 0 else "simultaneous")
    print(f"\n  Peak lag: {pk} min  →  {direction}")
    print(f"  Peak r  : {corr['peak_corr']:+.4f}")
    print(f"  p-value : {corr['peak_p']:.4f}")

    plot_correlogram(corr)

    has_lead_lag = bool(corr["peak_p"] < 0.05 and pk != 0)
    result = {
        "peak_lag":    corr["peak_lag"],
        "peak_corr":   corr["peak_corr"],
        "p_value":     corr["peak_p"],
        "significant": has_lead_lag,   # True only if non-zero lag is significant
        "direction":   direction,
        "n_windows":   int(corr["ns"][len(corr["lags"]) // 2]),
    }
    with open(DATA_DIR / "lead_lag_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved → data/lead_lag_result.json")

    if has_lead_lag:
        print(f"\n  Result: significant non-zero lag at {pk} min ({direction}) — "
              f"lagged ETH feature will be added in XGBoost.")
    else:
        print(f"\n  Result: peak at lag 0 (simultaneous movement, no lead-lag) — "
              f"XGBoost uses ETH P(UP) at lag 0. Concurrent correlation r = {corr['peak_corr']:.3f}.")


if __name__ == "__main__":
    main()
