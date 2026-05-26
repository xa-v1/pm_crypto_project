"""
Variance Ratio Test:  P(UP) vs log(spot)
========================================

Lo-MacKinlay (1988) variance ratio diagnoses departures from the
random-walk null:
    VR(q) = Var(x_t − x_{t−q}) / (q · Var(x_t − x_{t−1}))
    VR(q) = 1   →  series is a martingale (efficient forecaster)
    VR(q) > 1  →  positive autocorrelation: momentum / under-reaction
    VR(q) < 1  →  negative autocorrelation: mean-reversion / over-reaction

Two series are tested side-by-side at the same horizons:

  (1) P(UP) — pooled within-contract 1-min changes; contracts are 15 minutes
      so the maximum horizon is q = 7 (≈ half the contract length).
  (2) log(close) for BTC and ETH spot — continuous 1-min series.

For each series we report VR(q) at q ∈ {2, 3, 5, 7, 10}, plus a Hurst
exponent estimate from the slope of log Var(x_t − x_{t−q}) on log q.

CIs come from:
  - PM:   cluster bootstrap, resampling contracts.
  - Spot: stationary block bootstrap (Politis–Romano), block length ≈ √n.

Outputs
-------
data/variance_ratio.cleaned
make_figures/variance_ratio.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "make_figures"
FIGURES_DIR.mkdir(exist_ok=True)

HORIZONS = [2, 3, 5, 7, 10]
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
# Core: VR(q) on a sequence of contiguous chunks
# ---------------------------------------------------------------------------

def variance_ratio_from_chunks(chunks: list[np.ndarray], q: int) -> float | None:
    """
    Compute VR(q) by pooling 1-period and q-period (overlapping) changes
    across multiple contiguous chunks (e.g. one per Kalshi contract for PM,
    or a single chunk for continuous spot).
    """
    one_changes = []
    q_changes   = []
    for x in chunks:
        if len(x) < q + 1:
            continue
        one_changes.append(np.diff(x))
        q_changes.append(x[q:] - x[:-q])

    if not one_changes:
        return None

    r1 = np.concatenate(one_changes)
    rq = np.concatenate(q_changes)
    var1 = np.var(r1, ddof=1)
    varq = np.var(rq, ddof=1)
    if var1 <= 0:
        return None
    return float(varq / (q * var1))


def hurst_from_vr(vrs: dict[int, float]) -> float | None:
    """
    H from the slope of log Var_q on log q.  Since
        Var_q ≈ q^{2H} · Var_1   ⇒   log(VR(q)) ≈ (2H − 1) · log(q)
    H = 0.5 is the random-walk null; H > 0.5 long-memory; H < 0.5 mean-reverting.
    """
    items = [(q, v) for q, v in vrs.items() if v is not None and v > 0]
    if len(items) < 2:
        return None
    xs = np.array([np.log(q) for q, _ in items])
    ys = np.array([np.log(v) for _, v in items])
    slope, _ = np.polyfit(xs, ys, 1)
    return float(0.5 * (slope + 1.0))


# ---------------------------------------------------------------------------
# Two estimators with bootstrap CIs
# ---------------------------------------------------------------------------

def pm_variance_ratio(asset: str) -> dict:
    """Within-contract P(UP) — each contract is its own chunk."""
    df = pd.read_csv(DATA_DIR / f"kalshi_{asset.lower()}_prices.csv",
                     parse_dates=["Timestamp"])
    df = df.sort_values(["Market Ticker", "Timestamp"]).reset_index(drop=True)
    chunks = [g["P(UP)"].values for _, g in df.groupby("Market Ticker")]
    n_contracts = len(chunks)

    point = {q: variance_ratio_from_chunks(chunks, q) for q in HORIZONS}
    H_point = hurst_from_vr(point)

    boot_vr = {q: [] for q in HORIZONS}
    boot_H  = []
    chunks_arr = np.array(chunks, dtype=object)
    for _ in range(N_BOOTSTRAP):
        idx = RNG.integers(0, n_contracts, size=n_contracts)
        resampled = [chunks_arr[i] for i in idx]
        vrs = {q: variance_ratio_from_chunks(resampled, q) for q in HORIZONS}
        for q in HORIZONS:
            if vrs[q] is not None:
                boot_vr[q].append(vrs[q])
        H_b = hurst_from_vr(vrs)
        if H_b is not None:
            boot_H.append(H_b)

    ci = {q: (float(np.percentile(boot_vr[q], 2.5)),
              float(np.percentile(boot_vr[q], 97.5))) for q in HORIZONS}
    H_ci = (float(np.percentile(boot_H, 2.5)),
            float(np.percentile(boot_H, 97.5))) if boot_H else (np.nan, np.nan)

    return {
        "label":   f"PM:  P(UP) {asset}",
        "n_obs":   sum(len(c) for c in chunks),
        "n_units": n_contracts,
        "vr":      point,
        "vr_ci":   ci,
        "hurst":   H_point,
        "hurst_ci": H_ci,
    }


def spot_variance_ratio(asset: str) -> dict:
    """log(close) on the continuous 1-min spot series; block bootstrap."""
    df = pd.read_csv(DATA_DIR / f"spot_{asset.lower()}_1m.csv",
                     parse_dates=["open_time_utc"])
    df = df.sort_values("open_time_utc").reset_index(drop=True)
    x = np.log(df["close"].values)
    chunks = [x]
    n = len(x)

    point = {q: variance_ratio_from_chunks(chunks, q) for q in HORIZONS}
    H_point = hurst_from_vr(point)

    # Stationary block bootstrap (Politis–Romano). Expected block length L = √n.
    L = max(int(round(np.sqrt(n))), 30)
    boot_vr = {q: [] for q in HORIZONS}
    boot_H  = []
    for _ in range(N_BOOTSTRAP):
        # Build a length-n sample by stitching geometric-length blocks
        idx_buf = np.empty(n, dtype=np.int64)
        filled = 0
        while filled < n:
            start = int(RNG.integers(0, n))
            length = int(RNG.geometric(1.0 / L))
            take = min(length, n - filled)
            for j in range(take):
                idx_buf[filled + j] = (start + j) % n
            filled += take
        x_b = x[idx_buf]
        vrs = {q: variance_ratio_from_chunks([x_b], q) for q in HORIZONS}
        for q in HORIZONS:
            if vrs[q] is not None:
                boot_vr[q].append(vrs[q])
        H_b = hurst_from_vr(vrs)
        if H_b is not None:
            boot_H.append(H_b)

    ci = {q: (float(np.percentile(boot_vr[q], 2.5)),
              float(np.percentile(boot_vr[q], 97.5))) for q in HORIZONS}
    H_ci = (float(np.percentile(boot_H, 2.5)),
            float(np.percentile(boot_H, 97.5))) if boot_H else (np.nan, np.nan)

    return {
        "label":    f"Spot:  log(close) {asset}",
        "n_obs":    n,
        "n_units":  1,
        "vr":       point,
        "vr_ci":    ci,
        "hurst":    H_point,
        "hurst_ci": H_ci,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(results: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "PM:  P(UP) BTC":     "#4C78A8",
        "Spot:  log(close) BTC": "#1F4E79",
        "PM:  P(UP) ETH":     "#F58518",
        "Spot:  log(close) ETH": "#9C4A00",
    }
    markers = {
        "PM:  P(UP) BTC":         "o",
        "Spot:  log(close) BTC":  "s",
        "PM:  P(UP) ETH":         "o",
        "Spot:  log(close) ETH":  "s",
    }

    order = ["PM:  P(UP) BTC", "Spot:  log(close) BTC",
             "PM:  P(UP) ETH", "Spot:  log(close) ETH"]
    for label in order:
        if label not in results:
            continue
        res = results[label]
        xs = HORIZONS
        ys = [res["vr"][q] for q in xs]
        lo = [res["vr_ci"][q][0] for q in xs]
        hi = [res["vr_ci"][q][1] for q in xs]
        yerr = np.vstack([np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)])
        yerr = np.clip(yerr, 0, None)  # bootstrap CI vs point can be off by floating-point ε
        ax.errorbar(xs, ys, yerr=yerr, fmt=markers[label] + "-",
                    color=colors[label], capsize=4, linewidth=1.8,
                    markersize=8, label=f"{label}  (H={res['hurst']:.3f})")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2,
               label="Random-walk null  VR = 1")
    ax.set_xlabel("Horizon q (minutes)")
    ax.set_ylabel("Variance Ratio  VR(q)")
    ax.set_title(
        "Lo–MacKinlay Variance Ratio:  Is P(UP) a martingale?\n"
        "VR < 1 → mean-reverting (over-reaction).   "
        "VR > 1 → momentum (under-reaction)."
    )
    ax.set_xticks(HORIZONS)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out = FIGURES_DIR / "variance_ratio.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = {}
    for asset in ("BTC", "ETH"):
        print(f"\n=== {asset} PM ===")
        r = pm_variance_ratio(asset)
        results[r["label"]] = r
        print(f"  n_contracts={r['n_units']}, n_obs={r['n_obs']:,}")
        for q in HORIZONS:
            print(f"  VR(q={q:2d}) = {r['vr'][q]:>6.3f}  "
                  f"[{r['vr_ci'][q][0]:.3f}, {r['vr_ci'][q][1]:.3f}]")
        print(f"  Hurst H = {r['hurst']:.3f}  "
              f"[{r['hurst_ci'][0]:.3f}, {r['hurst_ci'][1]:.3f}]")

        print(f"\n=== {asset} Spot ===")
        r = spot_variance_ratio(asset)
        results[r["label"]] = r
        print(f"  n_obs={r['n_obs']:,}")
        for q in HORIZONS:
            print(f"  VR(q={q:2d}) = {r['vr'][q]:>6.3f}  "
                  f"[{r['vr_ci'][q][0]:.3f}, {r['vr_ci'][q][1]:.3f}]")
        print(f"  Hurst H = {r['hurst']:.3f}  "
              f"[{r['hurst_ci'][0]:.3f}, {r['hurst_ci'][1]:.3f}]")

    plot(results)

    dump = {label: {
        "vr":       {str(q): r["vr"][q] for q in HORIZONS},
        "vr_ci":    {str(q): list(r["vr_ci"][q]) for q in HORIZONS},
        "hurst":    r["hurst"],
        "hurst_ci": list(r["hurst_ci"]),
        "n_obs":    r["n_obs"],
        "n_units":  r["n_units"],
    } for label, r in results.items()}
    with open(DATA_DIR / "variance_ratio.cleaned", "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nSaved → {DATA_DIR / 'variance_ratio.cleaned'}")


if __name__ == "__main__":
    main()
