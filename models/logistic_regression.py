"""
Logistic Regression Baseline — BTC Kalshi 15-minute contracts.

Absorbs Matthew's scope (market disagreement signal + same-asset LR baseline)
and extends it with a full cross-asset feature set and comprehensive evaluation.

Why LR as baseline?
    Interpretable coefficients + p-values show *which* features drive predictions
    and *how confident* we are in each effect. LR is also a strong benchmark —
    if XGBoost can't beat it, the nonlinear complexity isn't justified.

Two models compared on the same matched dataset (~1,600 OOS contracts):
    Baseline LR — 3 same-asset features (Matthew's proposal scope)
    Extended LR — 12 features including new engineered features + cross-asset ETH

Loads : data/merged_contracts.csv
Saves : data/lr_baseline_oos.csv, data/lr_extended_oos.csv (for backtest.py)
Outputs:
    figures/disagreement_signal.png    — conviction spread vs realized accuracy
    figures/lr_coefficients.png        — standardized coefficient bar chart
    figures/lr_roc_pr.png              — ROC + precision-recall curves
    figures/lr_decision_boundary.png   — 2D classification boundary visualization
    figures/lr_calibration_heatmap.png — 2D heatmap of realized accuracy by features
"""

import warnings
# Suppress numerical overflow warnings from lbfgs during early gradient steps.
# Root cause: solver probes large step sizes before convergence; StandardScaler
# reduces this but early iterations can still trigger numpy overflow.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*non-interactive.*")

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Same-asset baseline — minimal interpretable model (Matthew's proposal scope)
BASELINE_FEATURES = ["opening_pup_btc", "momentum_3min_btc", "hour_of_open_btc"]

# Extended model — all engineered features + cross-asset signals.
# btc_eth_divergence is excluded here: it equals opening_pup_btc - opening_pup_eth,
# a perfect linear combination of two features already in the model. That makes
# the design matrix singular and produces NaN standard errors in statsmodels MLE.
# L2-regularized sklearn can still fit, but the coefficient table becomes unreadable.
# XGBoost (which is not linear) keeps all 12 features including btc_eth_divergence.
FULL_FEATURES = [
    "opening_pup_btc",       # market's initial probability assessment
    "momentum_1min_btc",     # first-minute drift (fastest available signal)
    "momentum_3min_btc",     # early directional drift
    "mean_pup_5min_btc",     # smoothed 5-min average, reduces opening noise
    "pup_vol_5min_btc",      # std of P(UP) in first 5 min — market uncertainty
    "conviction_spread_btc", # |P(UP) - 0.5| * 2: market one-sidedness at open
    "hour_of_open_btc",      # UTC hour — intraday liquidity patterns
    "volume_log_btc",        # market activity proxy (log1p of opening volume)
    "prev_open_pup_btc",     # previous contract's opening — market memory signal
    "opening_pup_eth",       # cross-asset directional signal at matched open
    "momentum_3min_eth",     # cross-asset early drift
]
TARGET = "target_btc"

FEATURE_LABELS = {
    "opening_pup_btc":       "BTC P(UP) at open",
    "momentum_1min_btc":     "BTC Momentum (1 min)",
    "momentum_3min_btc":     "BTC Momentum (3 min)",
    "mean_pup_5min_btc":     "BTC Mean P(UP) (5 min)",
    "pup_vol_5min_btc":      "BTC P(UP) Volatility (5 min)",
    "conviction_spread_btc": "BTC Conviction Spread",
    "hour_of_open_btc":      "Hour of Open (UTC)",
    "volume_log_btc":        "BTC Log Volume",
    "prev_open_pup_btc":     "BTC Prev. Open P(UP)",
    "opening_pup_eth":       "ETH P(UP) at open",
    "momentum_3min_eth":     "ETH Momentum (3 min)",
}

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "font.size":         11,
})


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    # Merged dataset ensures both Baseline and Extended LR are evaluated on
    # the same contracts — only way to make the model comparison fair.
    df = pd.read_csv(DATA_DIR / "merged_contracts.csv", parse_dates=["open_key"])
    df = df[df[TARGET].notna()].sort_values("open_key").reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    df = df[FULL_FEATURES + [TARGET, "open_key"]].dropna()
    print(f"Loaded {len(df):,} matched BTC/ETH contracts with outcomes.")
    print(f"Base rate (UP fraction): {df[TARGET].mean():.3f}")
    return df


# ---------------------------------------------------------------------------
# 1. Disagreement signal (Matthew's deliverable)
# ---------------------------------------------------------------------------

def plot_disagreement_signal(df: pd.DataFrame) -> None:
    """
    Test whether high-conviction markets (far from 50/50) are systematically
    over- or under-confident relative to realized accuracy.
    Conviction spread = |P(UP) - 0.5| * 2, range [0, 1].
    A well-calibrated market has bars that grow monotonically with conviction.
    """
    bins   = np.arange(0, 1.01, 0.10)
    labels = [f"{b:.1f}–{b + 0.1:.1f}" for b in bins[:-1]]
    df_plot = df.copy()
    df_plot["cs_bin"] = pd.cut(df_plot["conviction_spread_btc"], bins=bins,
                                labels=labels, right=False)

    stats = (
        df_plot.groupby("cs_bin", observed=True)[TARGET]
        .agg(accuracy="mean", n="count", std="std")
        .reset_index()
    )
    stats["ci95"] = 1.96 * stats["std"] / np.sqrt(stats["n"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(stats)), stats["accuracy"], yerr=stats["ci95"], capsize=4,
           color="steelblue", alpha=0.8, error_kw={"linewidth": 1.2},
           label="Realized accuracy")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="50% baseline")

    for i, row in stats.iterrows():
        ax.text(i, 0.32, f"n={int(row['n'])}", ha="center", fontsize=8, color="dimgray")

    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats["cs_bin"], rotation=40, ha="right", fontsize=9)
    ax.set_xlabel("Conviction Spread  |P(UP) − 0.5| × 2")
    ax.set_ylabel("Realized Accuracy")
    ax.set_title(
        "Market Disagreement Signal: Does Higher Conviction → Higher Accuracy?  (BTC)\n"
        "Bars above 0.5 = directional predictive value; error bars = 95% CI"
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0.28, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "disagreement_signal.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → figures/disagreement_signal.png")


# ---------------------------------------------------------------------------
# 2. Fit models with 5-fold time-series CV
# ---------------------------------------------------------------------------

def fit_models(df: pd.DataFrame) -> dict:
    """
    5-fold time-series CV for both LR models. TimeSeriesSplit always trains
    on the past and tests on the future — avoids any future data leakage
    that standard k-fold would introduce in sequential market data.
    """
    tscv = TimeSeriesSplit(n_splits=5)

    def make_lr():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0, random_state=42),
        )

    results = {}
    for name, features in [("Baseline LR", BASELINE_FEATURES), ("Extended LR", FULL_FEATURES)]:
        x = df[features]
        y = df[TARGET]

        oos_proba = np.full(len(x), np.nan)
        oos_pred  = np.full(len(x), -1, dtype=int)

        for train_idx, test_idx in tscv.split(x):
            pipe = make_lr()
            pipe.fit(x.iloc[train_idx], y.iloc[train_idx])
            oos_proba[test_idx] = pipe.predict_proba(x.iloc[test_idx])[:, 1]
            oos_pred[test_idx]  = pipe.predict(x.iloc[test_idx])

        mask  = oos_pred >= 0
        y_m   = y[mask]
        pred_m  = oos_pred[mask]
        proba_m = oos_proba[mask]

        results[name] = {
            "features":  features,
            "open_key":  df["open_key"].values[mask],
            "y":         y_m,
            "pred":      pred_m,
            "proba":     proba_m,
            "accuracy":  accuracy_score(y_m, pred_m),
            "precision": precision_score(y_m, pred_m, zero_division=0),
            "recall":    recall_score(y_m, pred_m, zero_division=0),
            "f1":        f1_score(y_m, pred_m, zero_division=0),
            "auc_roc":   roc_auc_score(y_m, proba_m),
        }

    return results


def save_oos_predictions(results: dict) -> None:
    """Save OOS probabilities so backtest.py can run all-model comparison."""
    for name, r in results.items():
        fname = "lr_baseline_oos.csv" if name == "Baseline LR" else "lr_extended_oos.csv"
        pd.DataFrame({
            "open_key":    r["open_key"],
            "target_btc":  r["y"].values,
            "lr_proba":    r["proba"],
        }).to_csv(DATA_DIR / fname, index=False)
        print(f"Saved → data/{fname}")


# ---------------------------------------------------------------------------
# 3. Statsmodels coefficient table
# ---------------------------------------------------------------------------

def print_statsmodels_summary(df: pd.DataFrame, features: list, label: str) -> None:
    """
    Full Logit fit on the entire dataset (not CV).
    Used for *coefficient interpretation* — p-values show statistical significance
    of each feature's directional association with outcome.
    Predictive power comes from the CV results, not this table.
    """
    x_const = sm.add_constant(df[features].values)
    model   = sm.Logit(df[TARGET], x_const).fit(disp=0)

    print(f"\n{'=' * 60}")
    print(f"  {label}  (statsmodels Logit, full dataset fit)")
    print(f"{'=' * 60}")
    print(f"  N = {len(df):,}   Base rate = {df[TARGET].mean():.3f}   "
          f"Pseudo R² (McFadden) = {model.prsquared:.4f}\n")

    params = model.params.values
    bse    = model.bse.values
    tvals  = model.tvalues.values
    pvals  = model.pvalues.values
    rows   = []
    for i, fname in enumerate(["const"] + features):
        rows.append({
            "Feature": FEATURE_LABELS.get(fname, fname),
            "Coef":    params[i],
            "Std Err": bse[i],
            "z-stat":  tvals[i],
            "p-value": pvals[i],
            "Sig":     ("***" if pvals[i] < 0.001 else
                        ("**"  if pvals[i] < 0.01  else
                        ("*"   if pvals[i] < 0.05  else ""))),
        })
    coef_df = pd.DataFrame(rows)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(coef_df.to_string(index=False))
    print("  Significance: * p<0.05  ** p<0.01  *** p<0.001")


# ---------------------------------------------------------------------------
# 4. Evaluation metric summary
# ---------------------------------------------------------------------------

def print_metrics(results: dict) -> None:
    print(f"\n{'=' * 60}")
    print("  5-Fold Time-Series CV — OOS Performance")
    print(f"{'=' * 60}")
    print(f"  {'Model':<18} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print(f"  {'-' * 57}")
    for name, r in results.items():
        print(f"  {name:<18} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['auc_roc']:>7.4f}")

    base_rate = results["Baseline LR"]["y"].mean()
    print(f"\n  Majority-class baseline accuracy: {max(base_rate, 1 - base_rate):.4f}")
    print(
        "\n  Interpretation:\n"
        "    Accuracy  — fraction of contracts where direction was predicted correctly\n"
        "    Precision — of predicted UPs, fraction that were actually UP\n"
        "    Recall    — of actual UPs, fraction the model caught\n"
        "    F1        — harmonic mean of precision and recall\n"
        "    AUC-ROC   — area under ROC; 0.5 = random, 1.0 = perfect"
    )


# ---------------------------------------------------------------------------
# 5. Standardized coefficient importance chart
# ---------------------------------------------------------------------------

def plot_coefficients(df: pd.DataFrame) -> None:
    """
    Coefficients from a StandardScaler → LR pipeline trained on all data.
    Magnitude is comparable across features because inputs are z-scored.
    A coefficient of +2 means a 1-SD increase in that feature raises the
    log-odds of an UP prediction by 2 units.
    """
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0, random_state=42),
    )
    pipe.fit(df[FULL_FEATURES], df[TARGET])
    coefs  = pipe.named_steps["logisticregression"].coef_[0]
    labels = [FEATURE_LABELS[f] for f in FULL_FEATURES]

    order        = np.argsort(np.abs(coefs))
    coefs_sorted = coefs[order]
    labels_sorted = [labels[i] for i in order]
    colors = ["tomato" if c > 0 else "steelblue" for c in coefs_sorted]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels_sorted, coefs_sorted, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized Coefficient  (log-odds per 1 SD)")
    ax.set_title(
        "Feature Importance — Extended Logistic Regression  (BTC)\n"
        "Red = increases P(UP prediction)   Blue = decreases P(UP prediction)"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lr_coefficients.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → figures/lr_coefficients.png")


# ---------------------------------------------------------------------------
# 6. ROC + Precision-Recall curves
# ---------------------------------------------------------------------------

def plot_roc_pr(results: dict) -> None:
    """
    ROC curve: how well does the model separate UP from DOWN across all thresholds?
    AUC > 0.5 = useful discriminative power above random.

    PR curve: especially informative with unequal base rates. A model that
    always predicts UP achieves recall=1 but precision=base_rate; the curve
    shows the full precision-recall tradeoff as the threshold varies.
    """
    colors = {"Baseline LR": "steelblue", "Extended LR": "tomato"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["y"], r["proba"])
        axes[0].plot(fpr, tpr, color=colors[name], linewidth=2,
                     label=f"{name}  (AUC = {r['auc_roc']:.3f})")

        precision, recall, _ = precision_recall_curve(r["y"], r["proba"])
        pr_auc = auc(recall, precision)
        axes[1].plot(recall, precision, color=colors[name], linewidth=2,
                     label=f"{name}  (AUC = {pr_auc:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve  (BTC, OOS)")
    axes[0].legend(loc="lower right", fontsize=9)

    base_rate = results["Baseline LR"]["y"].mean()
    axes[1].axhline(base_rate, color="gray", linestyle="--", linewidth=1,
                    alpha=0.6, label=f"Always-UP baseline ({base_rate:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve  (BTC, OOS)")
    axes[1].legend(loc="upper right", fontsize=9)

    plt.suptitle("LR Model Evaluation — BTC 15-min Kalshi Contracts", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lr_roc_pr.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → figures/lr_roc_pr.png")


# ---------------------------------------------------------------------------
# 7. Decision boundary visualization (2D slice)
# ---------------------------------------------------------------------------

def plot_decision_boundary(df: pd.DataFrame) -> None:
    """
    Classification boundary for the Extended LR model in the space of the two
    strongest predictors: momentum_3min_btc (z≈14, highest significance) and
    opening_pup_btc. All other features are held at their median.

    Coloring: predicted P(UP) from the trained model over a fine grid.
    Scatter: actual data points colored by true outcome (UP=blue, DOWN=red).
    Contour at 0.5 = the model's decision boundary.
    """
    feat_x = "momentum_3min_btc"
    feat_y = "opening_pup_btc"

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0, random_state=42),
    )
    pipe.fit(df[FULL_FEATURES], df[TARGET])

    # Build grid, fill non-plotted features at median
    x_range = np.linspace(df[feat_x].quantile(0.02), df[feat_x].quantile(0.98), 120)
    y_range = np.linspace(df[feat_y].quantile(0.02), df[feat_y].quantile(0.98), 120)
    xx, yy  = np.meshgrid(x_range, y_range)

    medians = df[FULL_FEATURES].median()
    grid_df = pd.DataFrame(
        np.tile(medians.values, (xx.size, 1)), columns=FULL_FEATURES
    )
    grid_df[feat_x] = xx.ravel()
    grid_df[feat_y] = yy.ravel()

    z = pipe.predict_proba(grid_df)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    cf = ax.contourf(xx, yy, z, levels=50, cmap="RdBu_r", alpha=0.75, vmin=0, vmax=1)
    plt.colorbar(cf, ax=ax, label="Predicted P(UP)")
    ax.contour(xx, yy, z, levels=[0.5], colors="black", linewidths=1.5,
               linestyles="--")

    up = df[df[TARGET] == 1]
    dn = df[df[TARGET] == 0]
    ax.scatter(up[feat_x], up[feat_y], c="steelblue", s=8, alpha=0.4, label="Actual UP")
    ax.scatter(dn[feat_x], dn[feat_y], c="tomato",    s=8, alpha=0.4, label="Actual DOWN")

    ax.set_xlabel(f"{FEATURE_LABELS[feat_x]}  ({feat_x})")
    ax.set_ylabel(f"{FEATURE_LABELS[feat_y]}  ({feat_y})")
    ax.set_title(
        "Extended LR — Decision Boundary  (BTC)\n"
        "2D slice through strongest predictors; all other features at median.\n"
        "Dashed black line = P(UP) = 0.5 classification boundary."
    )
    ax.legend(fontsize=9, markerscale=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lr_decision_boundary.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → figures/lr_decision_boundary.png")


# ---------------------------------------------------------------------------
# 8. 2D calibration heatmap
# ---------------------------------------------------------------------------

def plot_calibration_heatmap(df: pd.DataFrame) -> None:
    """
    2D heatmap: X = opening_pup_btc bins, Y = momentum_3min_btc bins.
    Cell color = fraction of contracts in that (pup, momentum) cell that ended UP.
    This reveals *where* in feature space the market is most predictable,
    and lets us spot regions where the signal is strongest or reversed.
    Cell count annotated to show data density.
    """
    pup_edges = np.linspace(df["opening_pup_btc"].quantile(0.02),
                            df["opening_pup_btc"].quantile(0.98), 8)
    mom_edges = np.linspace(df["momentum_3min_btc"].quantile(0.02),
                            df["momentum_3min_btc"].quantile(0.98), 8)

    df_plot = df[
        (df["opening_pup_btc"] >= pup_edges[0]) & (df["opening_pup_btc"] <= pup_edges[-1]) &
        (df["momentum_3min_btc"] >= mom_edges[0]) & (df["momentum_3min_btc"] <= mom_edges[-1])
    ].copy()

    df_plot["pup_bin"] = pd.cut(df_plot["opening_pup_btc"], bins=pup_edges, labels=False)
    df_plot["mom_bin"] = pd.cut(df_plot["momentum_3min_btc"], bins=mom_edges, labels=False)

    pivot_acc = df_plot.groupby(["mom_bin", "pup_bin"])[TARGET].mean().unstack("pup_bin")
    pivot_n   = df_plot.groupby(["mom_bin", "pup_bin"])[TARGET].count().unstack("pup_bin")

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot_acc.values, cmap="RdBu", vmin=0.3, vmax=0.8,
                   aspect="auto", origin="lower")
    plt.colorbar(im, ax=ax, label="Fraction Ending UP")

    n_pup = len(pup_edges) - 1
    n_mom = len(mom_edges) - 1
    for i in range(n_mom):
        for j in range(n_pup):
            try:
                acc_val = pivot_acc.iloc[i, j]
                cnt_val = pivot_n.iloc[i, j]
                if not np.isnan(acc_val):
                    text_color = "white" if abs(acc_val - 0.55) > 0.18 else "black"
                    ax.text(j, i, f"{acc_val:.2f}\nn={int(cnt_val)}",
                            ha="center", va="center", fontsize=8, color=text_color)
            except (IndexError, KeyError):
                pass

    pup_labels = [f"{v:.2f}" for v in (pup_edges[:-1] + pup_edges[1:]) / 2]
    mom_labels = [f"{v:+.3f}" for v in (mom_edges[:-1] + mom_edges[1:]) / 2]
    ax.set_xticks(range(n_pup))
    ax.set_xticklabels(pup_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_mom))
    ax.set_yticklabels(mom_labels, fontsize=9)
    ax.set_xlabel("Opening P(UP)")
    ax.set_ylabel("3-min Momentum  (P(UP)[min3] − P(UP)[min0])")
    ax.set_title(
        "Calibration Heatmap — Realized UP Rate by Feature Bin  (BTC)\n"
        "Blue = low UP rate, Red = high UP rate. Annotated with fraction and count."
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lr_calibration_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → figures/lr_calibration_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()

    print("\n--- 1. Disagreement Signal ---")
    plot_disagreement_signal(df)

    print("\n--- 2. Fitting LR models (5-fold time-series CV) ---")
    results = fit_models(df)

    print("\n--- 3. Coefficient tables (statsmodels, full-data fit) ---")
    print_statsmodels_summary(df, BASELINE_FEATURES, "Baseline LR (3 same-asset features)")
    print_statsmodels_summary(df, FULL_FEATURES,     "Extended LR (12 features)")

    print("\n--- 4. OOS evaluation metrics ---")
    print_metrics(results)

    print("\n--- 5. Saving OOS predictions for backtest.py ---")
    save_oos_predictions(results)

    print("\n--- 6. Generating figures ---")
    plot_coefficients(df)
    plot_roc_pr(results)
    plot_decision_boundary(df)
    plot_calibration_heatmap(df)

    print("\nDone. All figures saved to figures/.")


if __name__ == "__main__":
    main()
