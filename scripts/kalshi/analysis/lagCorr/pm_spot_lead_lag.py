"""
PM vs Spot Lead-Lag Cross-Correlogram
======================================

Tests whether Kalshi P(UP) leads, lags, or moves contemporaneously with the
underlying Coinbase spot price (BTC-USD and ETH-USD) at minute-level.

This is the test motivated by the whitepaper's §Applications: PM-vs-PM
lead-lagCorr (existing analysis/lead_lag.py) only proves the two PM markets share
a sentiment factor; it cannot establish whether that factor is incrementally
informative about *spot* prices.

Method
------
For each Kalshi minute-level observation at UTC time t inside contract c:
    dP(UP)_{c,t} = P(UP)_{c,t} - P(UP)_{c,t-1}        (within-contract change)
    r_{t+k}      = log(close_{t+k}) - log(close_{t+k-1})   (spot 1-min return)

At each lagCorr k ∈ {-10, ..., +10} we pool all valid (c, t) pairs and compute
the Pearson correlation between dP(UP)_{c,t} and r_{t+k}.

Sign convention
---------------
Lag k > 0 → P(UP) move at t aligns with spot return at t+k → **PM leads spot**.
Lag k < 0 → P(UP) move at t aligns with spot return at t-|k| → **spot leads PM**.
Lag k = 0 → contemporaneous.

Bootstrap CIs
-------------
Block-bootstrap resamples *contracts* (not individual minute pairs) so the
within-contract dependence structure is preserved.

Outputs
-------
data/pm_spot_lead_lag.cleaned  — peak lagCorr, peak r, and full per-lagCorr table
make_figures/pm_spot_lead_lag.png — two-panel correlogram (BTC, ETH)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "make_figures"
FIGURES_DIR.mkdir(exist_ok=True)

MAX_LAG = 10
N_BOOTSTRAP = 500
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "font.size":         11,
})


# ---------------------------------------------------------------------------
# Load & align minute-level Kalshi + spot data
# ---------------------------------------------------------------------------

def load_kalshi_with_utc(asset: str) -> pd.DataFrame:
    """Kalshi minute-level data with PT timestamps converted to UTC."""
    df = pd.read_csv(DATA_DIR / f"kalshi_{asset.lower()}_prices.csv",
                     parse_dates=["Timestamp"])
    # Original timestamps are in scraper local time (US/Pacific). Localize then
    # convert to UTC. ambiguous='infer' handles the DST transition (Mar 8 2026).
    df["ts_utc"] = (
        df["Timestamp"]
        .dt.tz_localize("America/Los_Angeles", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    df = df.sort_values(["Market Ticker", "ts_utc"]).reset_index(drop=True)
    df["minute_in_contract"] = df.groupby("Market Ticker").cumcount()
    df["dpup"] = df.groupby("Market Ticker")["P(UP)"].diff()
    # Round to minute so we can join cleanly against spot bars
    df["ts_utc_min"] = df["ts_utc"].dt.floor("min")
    return df[["Market Ticker", "minute_in_contract", "ts_utc_min", "P(UP)", "dpup"]]


def load_spot_with_returns(asset: str) -> pd.DataFrame:
    """Coinbase 1m bars with log-returns indexed on UTC open minute."""
    df = pd.read_csv(DATA_DIR / f"spot_{asset.lower()}_1m.csv",
                     parse_dates=["open_time_utc"])
    df = df.sort_values("open_time_utc").reset_index(drop=True)
    df["log_close"] = np.log(df["close"])
    df["ret_1m"]    = df["log_close"].diff()
    df["ts_utc_min"] = df["open_time_utc"].dt.floor("min")
    return df[["ts_utc_min", "close", "ret_1m"]]


def build_aligned_panel(asset: str) -> pd.DataFrame:
    """
    Return long-form panel with one row per (contract, minute) carrying:
        ticker, ts, dpup_t, ret_{t+k}  for k in [-MAX_LAG, +MAX_LAG]
    """
    kal  = load_kalshi_with_utc(asset)
    spot = load_spot_with_returns(asset)

    # Build a wide table of spot returns at offsets relative to the join time t.
    # ret_lag_p3 means r_{t+3}; ret_lag_m2 means r_{t-2}.
    spot = spot.set_index("ts_utc_min").sort_index()
    for k in range(-MAX_LAG, MAX_LAG + 1):
        # shift(-k) on the time-indexed series: row at time t pulls value from t+k
        spot[f"ret_lag_{k:+d}"] = spot["ret_1m"].shift(-k)
    spot = spot.reset_index()

    panel = kal.merge(spot, on="ts_utc_min", how="left")
    # Drop rows without a valid ΔP(UP) (first minute of each contract) and any
    # row whose corresponding spot bar is missing.
    panel = panel.dropna(subset=["dpup"])
    return panel


# ---------------------------------------------------------------------------
# Correlogram with cluster bootstrap
# ---------------------------------------------------------------------------

def correlogram_with_cis(panel: pd.DataFrame) -> dict:
    lags = list(range(-MAX_LAG, MAX_LAG + 1))
    tickers = panel["Market Ticker"].unique()

    point = {}
    for k in lags:
        col = f"ret_lag_{k:+d}"
        sub = panel[["dpup", col]].dropna()
        if len(sub) < 50:
            point[k] = (np.nan, len(sub))
            continue
        r = np.corrcoef(sub["dpup"].values, sub[col].values)[0, 1]
        point[k] = (float(r), int(len(sub)))

    # Cluster bootstrap over contracts
    boot_rs = {k: [] for k in lags}
    grouped = {t: panel[panel["Market Ticker"] == t] for t in tickers}
    for _ in range(N_BOOTSTRAP):
        sampled = RNG.choice(tickers, size=len(tickers), replace=True)
        boot = pd.concat([grouped[t] for t in sampled], ignore_index=True, copy=False)
        for k in lags:
            col = f"ret_lag_{k:+d}"
            sub = boot[["dpup", col]].dropna()
            if len(sub) >= 50:
                r = np.corrcoef(sub["dpup"].values, sub[col].values)[0, 1]
                boot_rs[k].append(r)

    cis = {k: (np.nanpercentile(boot_rs[k], 2.5),
               np.nanpercentile(boot_rs[k], 97.5)) for k in lags}

    # Peak by |r|
    valid_lags = [k for k in lags if not np.isnan(point[k][0])]
    peak_lag = max(valid_lags, key=lambda k: abs(point[k][0]))

    return {
        "lags":     lags,
        "r":        {k: point[k][0] for k in lags},
        "n":        {k: point[k][1] for k in lags},
        "ci_lo":    {k: cis[k][0] for k in lags},
        "ci_hi":    {k: cis[k][1] for k in lags},
        "peak_lag": peak_lag,
        "peak_r":   point[peak_lag][0],
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_two_panel(results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    colors = {"BTC": "steelblue", "ETH": "darkorange"}

    for ax, asset in zip(axes, ("BTC", "ETH")):
        res = results[asset]
        lags = res["lags"]
        rs   = [res["r"][k] for k in lags]
        lo   = [res["ci_lo"][k] for k in lags]
        hi   = [res["ci_hi"][k] for k in lags]
        yerr = np.vstack([np.array(rs) - np.array(lo), np.array(hi) - np.array(rs)])

        ax.bar(lags, rs, color=colors[asset], alpha=0.8, width=0.7,
               label="Mean Pearson r")
        ax.errorbar(lags, rs, yerr=yerr, fmt="none", color="black",
                    capsize=3, linewidth=1.0, label="95% cluster-bootstrap CI")
        # Mark CI-significant lags (CI excludes zero)
        for i, k in enumerate(lags):
            if lo[i] > 0 or hi[i] < 0:
                ax.text(k, rs[i] + np.sign(rs[i]) * 0.005,
                        "*", ha="center", va="bottom", fontsize=11)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xlabel("Lag k (minutes)  →   k > 0:  PM leads spot")
        ax.set_xticks(range(-MAX_LAG, MAX_LAG + 1, 2))
        ax.set_title(
            f"{asset}:  corr( ΔP(UP)_t , r_spot_{{t+k}} )\n"
            f"peak |r| at k = {res['peak_lag']} min  (r = {res['peak_r']:+.4f})",
            fontsize=11,
        )
        if asset == "BTC":
            ax.set_ylabel("Pearson r")
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "PM → Spot Lead-Lag Cross-Correlogram\n"
        "Within-contract ΔP(UP) vs. concurrent and lagged 1-min spot log-returns",
        y=1.02, fontsize=12,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "pm_spot_lead_lag.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = {}
    for asset in ("BTC", "ETH"):
        print(f"\n=== {asset} ===")
        panel = build_aligned_panel(asset)
        print(f"  Aligned (contract, minute) rows: {len(panel):,}")
        print(f"  Unique contracts            : {panel['Market Ticker'].nunique():,}")

        res = correlogram_with_cis(panel)
        results[asset] = res

        print(f"  {'lagCorr':>4} {'r':>9} {'CI95':>22}  {'n':>7}")
        for k in res["lags"]:
            print(f"  {k:>4d} {res['r'][k]:>9.4f}  "
                  f"[{res['ci_lo'][k]:>+.4f}, {res['ci_hi'][k]:>+.4f}]  "
                  f"{res['n'][k]:>7,}")
        print(f"  peak: lagCorr = {res['peak_lag']} min, r = {res['peak_r']:+.4f}")

    plot_two_panel(results)

    # Pure-Python dump
    dump = {a: {"lags": r["lags"],
                "r":    [r["r"][k] for k in r["lags"]],
                "ci_lo":[r["ci_lo"][k] for k in r["lags"]],
                "ci_hi":[r["ci_hi"][k] for k in r["lags"]],
                "n":    [r["n"][k] for k in r["lags"]],
                "peak_lag": int(r["peak_lag"]),
                "peak_r":   float(r["peak_r"])} for a, r in results.items()}
    out = DATA_DIR / "pm_spot_lead_lag.cleaned"
    with open(out, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
