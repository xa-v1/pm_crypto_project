"""
Incremental Information Test:  PM features vs. Spot baseline
=============================================================

Question
--------
Do Kalshi P(UP) features add predictive power above what spot price features
already contain?  Three nested analysis are fit on the *same* matched contracts:

  (a) SPOT-only — features built from BTC spot price in the same 0-to-5-min
                   contract window the PM features use.
  (b) PM-only   — the existing FULL_FEATURES set from logistic_regression.py.
  (c) COMBINED  — union of the two feature sets.

All three are evaluated with 5-fold time-series CV (folds ordered
chronologically) and reported with bootstrap CIs and a DeLong AUC test.

Spot features (mirroring the PM 0-to-5-min lookback)
----------------------------------------------------
  spot_ret_1m       log return over the 1 min ending at contract open
  spot_ret_3m       log return over the prior 3 min ending at contract open
  spot_ret_5m       log return over the prior 5 min ending at contract open
  spot_mom_3m_post  log return from contract minute 0 → 3
  spot_mom_5m_post  log return from contract minute 0 → 5
  spot_rv_5m_pre    std of 1-min log returns in prior 5 min ending at open
  spot_rv_5m_post   std of 1-min log returns in contract minutes 0..5
  spot_vol_log_5m   log(1 + summed spot volume in prior 5 min)

The PM feature set uses contract minutes 0..5 (per the existing pipeline), so
both the pre- and post-open spot windows are included for a fair test: spot
"knows" the same minute window the PM features were built on.

Loads
-----
data/merged_contracts.csv  — PM features + outcome (per existing pipeline)
data/spot_btc_1m.csv       — Coinbase BTC 1m bars (already UTC)
data/kalshi_btc_prices.csv — only used to confirm timestamp tz

Outputs
-------
data/incremental_oos.csv     — OOS predictions for all three analysis
data/incremental_summary.cleaned — metrics + DeLong p-values + bootstrap CIs
make_figures/incremental_information.png  — bar chart + ROC overlay
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "make_figures"
FIGURES_DIR.mkdir(exist_ok=True)

TARGET = "target_btc"
RNG = np.random.default_rng(42)
N_BOOTSTRAP = 1000

PM_FEATURES = [
    "opening_pup_btc",
    "momentum_1min_btc",
    "momentum_3min_btc",
    "mean_pup_5min_btc",
    "pup_vol_5min_btc",
    "conviction_spread_btc",
    "hour_of_open_btc",
    "volume_log_btc",
    "prev_open_pup_btc",
]
SPOT_FEATURES = [
    "spot_ret_1m_pre",
    "spot_ret_3m_pre",
    "spot_ret_5m_pre",
    "spot_mom_3m_post",
    "spot_mom_5m_post",
    "spot_rv_5m_pre",
    "spot_rv_5m_post",
    "spot_vol_log_5m_pre",
]

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "font.size":         11,
})


# ---------------------------------------------------------------------------
# Spot feature construction
# ---------------------------------------------------------------------------

def build_spot_features(merged: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    """
    For each row in merged_contracts.csv, compute spot features using the
    1-min BTC spot bars indexed by UTC. contract_open_btc is in US/Pacific
    (per scraper convention); we convert to UTC for the spot join.
    """
    spot = spot.sort_values("open_time_utc").reset_index(drop=True)
    spot["log_close"] = np.log(spot["close"])
    spot["ret_1m"]    = spot["log_close"].diff()
    spot["ts_min"]    = spot["open_time_utc"].dt.floor("min")
    spot = spot.set_index("ts_min")

    merged = merged.copy()
    merged["contract_open_utc"] = (
        pd.to_datetime(merged["contract_open_btc"])
          .dt.tz_localize("America/Los_Angeles",
                          ambiguous="infer", nonexistent="shift_forward")
          .dt.tz_convert("UTC")
          .dt.floor("min")
    )

    feats = []
    for t0 in merged["contract_open_utc"]:
        # Pre-open windows
        try:
            close_t0      = spot.at[t0, "log_close"]
            close_t0_m1   = spot.at[t0 - pd.Timedelta("1min"), "log_close"]
            close_t0_m3   = spot.at[t0 - pd.Timedelta("3min"), "log_close"]
            close_t0_m5   = spot.at[t0 - pd.Timedelta("5min"), "log_close"]
            close_t0_p3   = spot.at[t0 + pd.Timedelta("3min"), "log_close"]
            close_t0_p5   = spot.at[t0 + pd.Timedelta("5min"), "log_close"]

            pre_window  = spot.loc[t0 - pd.Timedelta("5min"): t0 - pd.Timedelta("1min")]
            post_window = spot.loc[t0 + pd.Timedelta("1min"): t0 + pd.Timedelta("5min")]
            pre_vol     = pre_window["volume"].sum()

            feats.append({
                "spot_ret_1m_pre":     close_t0 - close_t0_m1,
                "spot_ret_3m_pre":     close_t0 - close_t0_m3,
                "spot_ret_5m_pre":     close_t0 - close_t0_m5,
                "spot_mom_3m_post":    close_t0_p3 - close_t0,
                "spot_mom_5m_post":    close_t0_p5 - close_t0,
                "spot_rv_5m_pre":      pre_window["ret_1m"].std(),
                "spot_rv_5m_post":     post_window["ret_1m"].std(),
                "spot_vol_log_5m_pre": float(np.log1p(pre_vol)),
            })
        except KeyError:
            # Missing spot bar at a needed minute
            feats.append({c: np.nan for c in SPOT_FEATURES})

    return pd.concat([merged.reset_index(drop=True), pd.DataFrame(feats)], axis=1)


# ---------------------------------------------------------------------------
# Cross-validated probabilities
# ---------------------------------------------------------------------------

def cv_proba(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    tscv = TimeSeriesSplit(n_splits=5)
    x = df[features].values
    y = df[TARGET].values
    oos = np.full(len(df), np.nan)
    for train_idx, test_idx in tscv.split(x):
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0, random_state=42),
        )
        pipe.fit(x[train_idx], y[train_idx])
        oos[test_idx] = pipe.predict_proba(x[test_idx])[:, 1]
    return oos


# ---------------------------------------------------------------------------
# DeLong test for paired AUC differences
# ---------------------------------------------------------------------------
# Standard DeLong (1988) implementation; small enough to inline here.

def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    final = np.empty(n)
    final[order] = ranks
    return final


def _delong_components(probs1, probs0):
    n1, n0 = len(probs1), len(probs0)
    m = n1 + n0
    Z = np.concatenate([probs1, probs0])
    tz = _midrank(Z)
    tx = _midrank(probs1)
    ty = _midrank(probs0)
    auc = (tz[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    v01 = (tz[:n1] - tx) / n0
    v10 = 1 - (tz[n1:] - ty) / n1
    return auc, v01, v10


def delong_test(y, p_a, p_b) -> tuple[float, float, float]:
    y = np.asarray(y).astype(bool)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    pos_a, neg_a = p_a[y], p_a[~y]
    pos_b, neg_b = p_b[y], p_b[~y]

    auc_a, v01_a, v10_a = _delong_components(pos_a, neg_a)
    auc_b, v01_b, v10_b = _delong_components(pos_b, neg_b)

    n1, n0 = pos_a.size, neg_a.size
    s01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1)
    s10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1)
    var = (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n1 + \
          (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / n0
    if var <= 0:
        return auc_a, auc_b, float("nan")
    z = (auc_a - auc_b) / np.sqrt(var)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return auc_a, auc_b, float(p)


# ---------------------------------------------------------------------------
# Bootstrap CIs (cluster-free; standard percentile bootstrap on contracts)
# ---------------------------------------------------------------------------

def bootstrap_metric(y, p, metric_fn, n_boot=N_BOOTSTRAP) -> tuple[float, float]:
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    samples = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        try:
            samples.append(metric_fn(y[idx], p[idx]))
        except ValueError:
            continue
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(metrics: dict, oos: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ----- (1) Bar chart of AUC with bootstrap CI -----
    names = ["Spot-only", "PM-only", "Combined"]
    aucs  = [metrics[n]["auc"] for n in names]
    los   = [metrics[n]["auc_ci"][0] for n in names]
    his   = [metrics[n]["auc_ci"][1] for n in names]
    errs  = np.vstack([np.array(aucs) - np.array(los), np.array(his) - np.array(aucs)])
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    bars = axes[0].bar(names, aucs, color=colors, alpha=0.85,
                       yerr=errs, capsize=6, error_kw={"linewidth": 1.4})
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7,
                    label="Random (AUC = 0.5)")
    for b, v in zip(bars, aucs):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                     f"{v:.3f}", ha="center", fontsize=10)
    axes[0].set_ylabel("AUC-ROC  (95% bootstrap CI)")
    axes[0].set_title("Out-of-Sample AUC by Feature Set")
    axes[0].set_ylim(0.45, max(his) + 0.05)
    axes[0].legend(loc="lower right")

    # ----- (2) ROC overlay -----
    for name, color in zip(names, colors):
        fpr, tpr, _ = roc_curve(oos["y"], oos[name])
        axes[1].plot(fpr, tpr, color=color, linewidth=2,
                     label=f"{name}  (AUC = {metrics[name]['auc']:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curves  (OOS)")
    axes[1].legend(loc="lower right", fontsize=9)

    # Annotate DeLong p-values below the plot title area
    pv_combined_vs_spot = metrics["delong"]["combined_vs_spot"]
    pv_combined_vs_pm   = metrics["delong"]["combined_vs_pm"]
    pv_pm_vs_spot       = metrics["delong"]["pm_vs_spot"]
    fig.suptitle(
        "Incremental Information Test — BTC 15-min direction\n"
        f"DeLong AUC: combined vs spot  p = {pv_combined_vs_spot:.3f}    "
        f"combined vs PM  p = {pv_combined_vs_pm:.3f}    "
        f"PM vs spot  p = {pv_pm_vs_spot:.3f}",
        y=1.05, fontsize=11,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "incremental_information.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading merged contracts and spot data...")
    merged = pd.read_csv(DATA_DIR / "merged_contracts.csv")
    merged = merged.dropna(subset=[TARGET]).reset_index(drop=True)
    merged[TARGET] = merged[TARGET].astype(int)

    spot = pd.read_csv(DATA_DIR / "spot_btc_1m.csv", parse_dates=["open_time_utc"])

    print("Building spot features at contract-level...")
    df = build_spot_features(merged, spot)
    before = len(df)
    df = df.dropna(subset=PM_FEATURES + SPOT_FEATURES + [TARGET]).reset_index(drop=True)
    df = df.sort_values("contract_open_utc").reset_index(drop=True)
    print(f"  Rows usable: {len(df):,}  (dropped {before - len(df):,} for missing features)")
    print(f"  Base rate (UP): {df[TARGET].mean():.3f}")

    print("\nFitting analysis (5-fold time-series CV)...")
    p_spot = cv_proba(df, SPOT_FEATURES)
    p_pm   = cv_proba(df, PM_FEATURES)
    p_comb = cv_proba(df, SPOT_FEATURES + PM_FEATURES)

    mask = ~(np.isnan(p_spot) | np.isnan(p_pm) | np.isnan(p_comb))
    y = df[TARGET].values[mask]
    p_spot, p_pm, p_comb = p_spot[mask], p_pm[mask], p_comb[mask]

    print(f"  OOS observations: {len(y):,}")

    metrics = {}
    for name, p in [("Spot-only", p_spot), ("PM-only", p_pm), ("Combined", p_comb)]:
        auc = roc_auc_score(y, p)
        acc = accuracy_score(y, (p > 0.5).astype(int))
        auc_ci = bootstrap_metric(y, p, roc_auc_score)
        acc_ci = bootstrap_metric(y, p, lambda yt, pt: accuracy_score(yt, (pt > 0.5).astype(int)))
        metrics[name] = {"auc": auc, "auc_ci": auc_ci, "accuracy": acc, "acc_ci": acc_ci,
                         "n": int(len(y))}

    auc_spot_, auc_comb_, p_cs = delong_test(y, p_comb, p_spot)
    auc_pm_,   auc_comb2, p_cp = delong_test(y, p_comb, p_pm)
    auc_pm__,  auc_spot2, p_ps = delong_test(y, p_pm,   p_spot)
    metrics["delong"] = {
        "combined_vs_spot": p_cs,
        "combined_vs_pm":   p_cp,
        "pm_vs_spot":       p_ps,
    }

    print("\n" + "=" * 64)
    print(f"{'Model':<14} {'AUC':>8}  {'AUC 95% CI':>22}  {'Acc':>8}  {'Acc 95% CI':>22}")
    print("-" * 64)
    for n in ("Spot-only", "PM-only", "Combined"):
        m = metrics[n]
        print(f"{n:<14} {m['auc']:>8.4f}  "
              f"[{m['auc_ci'][0]:>.4f},{m['auc_ci'][1]:>.4f}]   "
              f"{m['accuracy']:>8.4f}  "
              f"[{m['acc_ci'][0]:>.4f},{m['acc_ci'][1]:>.4f}]")
    print("=" * 64)
    print("DeLong tests:")
    print(f"  Combined vs Spot-only  ΔAUC = {metrics['Combined']['auc']-metrics['Spot-only']['auc']:+.4f}  p = {p_cs:.4f}")
    print(f"  Combined vs PM-only    ΔAUC = {metrics['Combined']['auc']-metrics['PM-only']['auc']:+.4f}  p = {p_cp:.4f}")
    print(f"  PM-only  vs Spot-only  ΔAUC = {metrics['PM-only']['auc']-metrics['Spot-only']['auc']:+.4f}  p = {p_ps:.4f}")

    # Save OOS predictions
    oos = pd.DataFrame({"y": y, "Spot-only": p_spot, "PM-only": p_pm, "Combined": p_comb})
    oos.to_csv(DATA_DIR / "incremental_oos.csv", index=False)
    print(f"\nSaved → {DATA_DIR / 'incremental_oos.csv'}")

    with open(DATA_DIR / "incremental_summary.cleaned", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"Saved → {DATA_DIR / 'incremental_summary.cleaned'}")

    plot_results(metrics, oos)


if __name__ == "__main__":
    main()
