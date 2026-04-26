"""
XGBoost classifier with full cross-asset feature set.
5-fold time-series CV. SHAP analysis. Comparison vs same-asset LR baseline.

Reads lead_lag_result.json (from lead_lag.py) to decide whether to include
a lagged ETH feature. Falls back gracefully if the file doesn't exist.

Loads: data/merged_contracts.csv, data/lead_lag_result.json (optional)
Outputs:
    figures/xgb_shap_bar.png        — mean |SHAP| per feature (importance ranking)
    figures/xgb_shap_beeswarm.png   — SHAP beeswarm (direction + magnitude)
    figures/model_comparison.png    — accuracy + AUC: Baseline LR vs XGB
    data/xgb_oos_predictions.csv    — OOS probabilities for backtest.py
"""

import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*non-interactive.*")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Full feature set — matches the Extended LR in logistic_regression.py.
# XGBoost handles multicollinearity natively (tree splits are non-linear),
# so all features are included; SHAP will reveal which ones actually matter.
BASE_FEATURES = [
    "opening_pup_btc",
    "momentum_1min_btc",
    "momentum_3min_btc",
    "mean_pup_5min_btc",
    "pup_vol_5min_btc",
    "conviction_spread_btc",
    "hour_of_open_btc",
    "volume_log_btc",
    "prev_open_pup_btc",
    "opening_pup_eth",
    "momentum_3min_eth",
    "btc_eth_divergence",
]

# Same 3-feature baseline used in logistic_regression.py — gives a fair
# comparison against the minimal LR model on the same OOS folds.
LR_FEATURES = ["opening_pup_btc", "momentum_3min_btc", "hour_of_open_btc"]
TARGET = "target_btc"

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "font.size":         11,
})


# ---------------------------------------------------------------------------
# Data + optional lagged ETH feature
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA_DIR / "merged_contracts.csv", parse_dates=["open_key"])
    df = df[df[TARGET].notna()].sort_values("open_key").reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)

    features = BASE_FEATURES.copy()

    # Add a shifted ETH P(UP) feature only if lead_lag.py found a significant
    # non-zero lag. Lag=0 (simultaneous movement, r=0.78) is already captured
    # by opening_pup_eth; a lagged version is only useful when ETH genuinely leads.
    lag_path = DATA_DIR / "lead_lag_result.json"
    if lag_path.exists():
        with open(lag_path) as f:
            ll = json.load(f)
        peak_lag = ll["peak_lag"]
        if ll["significant"] and peak_lag != 0:
            col = f"eth_pup_lag{abs(peak_lag)}"
            shift_by = abs(peak_lag) if peak_lag < 0 else -peak_lag
            df[col] = df["opening_pup_eth"].shift(shift_by)
            features.append(col)
            print(f"  Added lagged ETH feature '{col}' (peak lag = {peak_lag} min)")
        else:
            print(f"  Lead-lag: peak at lag 0 (concurrent, r={ll['peak_corr']:.3f}); "
                  f"no lagged feature added.")
    else:
        print("  lead_lag_result.json not found; skipping lagged ETH feature.")

    df = df[features + [TARGET, "open_key"]].dropna()
    return df, features


# ---------------------------------------------------------------------------
# 5-fold time-series CV: XGBoost + Baseline LR
# ---------------------------------------------------------------------------

def run_cv(df: pd.DataFrame, features: list[str]) -> dict:
    x = df[features]
    y = df[TARGET]
    tscv = TimeSeriesSplit(n_splits=5)

    # Conservative hyperparameters to reduce overfitting on small-ish dataset:
    # max_depth=4 keeps trees shallow; min_child_weight=5 prevents tiny leaves.
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    lr_pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=42),
    )

    xgb_proba = np.full(len(x), np.nan)
    xgb_pred  = np.full(len(x), -1, dtype=int)
    lr_pred   = np.full(len(x), -1, dtype=int)
    lr_proba  = np.full(len(x), np.nan)

    print("\n5-Fold Time-Series CV:")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(x), 1):
        x_tr, x_te = x.iloc[train_idx], x.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        xgb_model.fit(x_tr, y_tr)
        xgb_proba[test_idx] = xgb_model.predict_proba(x_te)[:, 1]
        xgb_pred[test_idx]  = xgb_model.predict(x_te)

        lr_pipe.fit(x_tr[LR_FEATURES], y_tr)
        lr_proba[test_idx] = lr_pipe.predict_proba(x_te[LR_FEATURES])[:, 1]
        lr_pred[test_idx]  = lr_pipe.predict(x_te[LR_FEATURES])

        fold_acc = accuracy_score(y_te, xgb_model.predict(x_te))
        print(f"  Fold {fold}: XGB acc = {fold_acc:.4f}  (n_test = {len(y_te):,})")

    mask = xgb_pred >= 0
    y_m  = y[mask]
    return {
        "xgb_acc":   accuracy_score(y_m, xgb_pred[mask]),
        "xgb_auc":   roc_auc_score(y_m, xgb_proba[mask]),
        "lr_acc":    accuracy_score(y_m, lr_pred[mask]),
        "lr_auc":    roc_auc_score(y_m, lr_proba[mask]),
        "xgb_pred":  xgb_pred,
        "xgb_proba": xgb_proba,
        "lr_pred":   lr_pred,
        "mask":      mask,
        "y":         y,
    }


def mcnemar_test(y_true, xgb_pred, lr_pred, mask) -> tuple[float, float]:
    """
    Edwards continuity-corrected McNemar test.
    Tests whether XGBoost and Baseline LR make *different* errors — not just
    whether one has higher accuracy. Significant result (p<0.05) means the two
    models are disagreeing in a non-random way.
    """
    y_m   = y_true[mask].values
    xgb_m = xgb_pred[mask]
    lr_m  = lr_pred[mask]
    n10 = int(np.sum((xgb_m == y_m) & (lr_m != y_m)))  # XGB right, LR wrong
    n01 = int(np.sum((xgb_m != y_m) & (lr_m == y_m)))  # LR right, XGB wrong
    # chi2 with continuity correction
    chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01) if (n10 + n01) > 0 else 0.0
    from scipy.stats import chi2 as chi2_dist
    p = float(chi2_dist.sf(chi2, df=1))
    return float(chi2), p


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_model_comparison(results: dict, chi2: float, p_mcnemar: float) -> None:
    """
    Side-by-side accuracy and AUC bars: Baseline LR vs XGBoost.
    McNemar p-value answers whether their difference is statistically significant.
    """
    metrics  = ["Accuracy", "AUC-ROC"]
    lr_vals  = [results["lr_acc"],  results["lr_auc"]]
    xgb_vals = [results["xgb_acc"], results["xgb_auc"]]
    x     = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_lr  = ax.bar(x - width / 2, lr_vals,  width, label="Same-Asset LR",
                      color="steelblue", alpha=0.85)
    bars_xgb = ax.bar(x + width / 2, xgb_vals, width, label="Cross-Asset XGBoost",
                      color="tomato",    alpha=0.85)

    for bar in bars_lr:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_xgb:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
               label="50% baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0.45, 0.80)
    sig_str = f"p = {p_mcnemar:.3f}" + (" *" if p_mcnemar < 0.05 else "")
    ax.set_title(
        f"Model Comparison — 5-Fold Time-Series CV  (BTC)\n"
        f"McNemar test: {sig_str}"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → figures/model_comparison.png")


def plot_shap(model: xgb.XGBClassifier, x: pd.DataFrame) -> None:
    """
    SHAP (SHapley Additive exPlanations) reveals which features actually drive
    each individual prediction, not just global importance. TreeExplainer is
    exact for tree-based models (no approximation needed).
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(x)

    # Bar: mean absolute SHAP value per feature — global importance ranking
    shap.summary_plot(sv, x, plot_type="bar", show=False)
    plt.title("XGBoost Feature Importance  (mean |SHAP value|)", pad=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "xgb_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → figures/xgb_shap_bar.png")

    # Beeswarm: each dot = one contract; color = feature value; x = SHAP impact.
    # Shows both direction and magnitude of each feature's effect on individual predictions.
    shap.summary_plot(sv, x, show=False)
    plt.title("XGBoost SHAP Beeswarm  (impact on model output)", pad=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "xgb_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → figures/xgb_shap_beeswarm.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    df, features = load_data()
    print(f"  Matched contracts (both outcomes): {len(df):,}")
    print(f"  Features ({len(features)}): {features}")

    results = run_cv(df, features)
    chi2, p_mc = mcnemar_test(results["y"], results["xgb_pred"],
                               results["lr_pred"], results["mask"])

    print("\n" + "=" * 52)
    print("OOS Performance Summary")
    print("=" * 52)
    print(f"  Same-Asset LR  | Acc: {results['lr_acc']:.4f}  | AUC: {results['lr_auc']:.4f}")
    print(f"  XGBoost        | Acc: {results['xgb_acc']:.4f}  | AUC: {results['xgb_auc']:.4f}")
    print(f"  McNemar test   | χ² = {chi2:.3f}  p = {p_mc:.4f}"
          + ("  (significant)" if p_mc < 0.05 else "  (not significant)"))

    plot_model_comparison(results, chi2, p_mc)

    # Final model trained on all data for SHAP — using all data gives the most
    # stable feature importance estimates (not bound by any single fold's train set)
    print("\nFitting final model on full dataset for SHAP analysis...")
    x_all = df[features]
    y_all = df[TARGET]
    final_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    final_model.fit(x_all, y_all)
    plot_shap(final_model, x_all)

    # Save OOS predictions aligned with open_key for backtest.py
    mask   = results["mask"]
    oos_df = df[mask][["open_key", TARGET]].copy()
    oos_df["xgb_proba"] = results["xgb_proba"][mask]
    oos_df["xgb_pred"]  = results["xgb_pred"][mask]
    oos_df.to_csv(DATA_DIR / "xgb_oos_predictions.csv", index=False)
    print("Saved → data/xgb_oos_predictions.csv")


if __name__ == "__main__":
    main()
