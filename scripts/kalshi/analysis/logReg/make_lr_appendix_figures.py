"""
Appendix figures for the incremental-information LR (slide 13 deep dive).

Fits the three nested logistic regressions used in incremental_information.py
(Spot-only / Kalshi-only / Combined) on the full panel, then produces two
diagnostic figures for the presentation appendix:

  - figures/logReg/feature_importance.png
        Standardized |coefficient| per feature in each LR, grouped bars.
        Highlights which features carry the directional signal in each model
        and why the Combined model is dominated by spot features.

  - figures/logReg/decision_boundary.png
        2D projection of the Combined LR's decision boundary on
        (spot_ret_5m_pre × opening_pup_btc). Other features held at their
        sample medians. Background shading is the model's predicted P(UP);
        dots are observed contracts colored by realized outcome.

Reads : data/cleaned/merged_contracts.csv + data/btc/spot_btc_1m.csv
Outputs: figures/logReg/feature_importance.png, decision_boundary.png
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Reuse the same spot-feature pipeline used by the headline test
from scripts.kalshi.analysis.logReg.incremental_information import (
    build_spot_features,
    SPOT_FEATURES as _SPOT_FEATURES_FROM_TEST,
    PM_FEATURES  as _KALSHI_FEATURES_FROM_TEST,
)

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
})

ROOT       = Path(__file__).resolve().parents[4]
MERGED     = ROOT / "data" / "cleaned" / "merged_contracts.csv"
SPOT_CSV   = ROOT / "data" / "btc"     / "spot_btc_1m.csv"
FIG_DIR    = ROOT / "figures" / "logReg"
TARGET     = "target_btc"

# Feature sets imported from incremental_information.py so they stay in sync.
KALSHI_FEATURES = list(_KALSHI_FEATURES_FROM_TEST)
SPOT_FEATURES   = list(_SPOT_FEATURES_FROM_TEST)


# ---------------------------------------------------------------------------
# Fit + extract standardized coefficients
# ---------------------------------------------------------------------------

def fit_and_get_coefs(df: pd.DataFrame, features: list[str]) -> pd.Series:
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0, random_state=42),
    )
    pipe.fit(df[features].values, df[TARGET].values)
    coef = pipe.named_steps["logisticregression"].coef_[0]
    return pd.Series(coef, index=features)


# ---------------------------------------------------------------------------
# Figure 1 — feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(coefs_by_model: dict[str, pd.Series]) -> None:
    all_features = []
    for s in coefs_by_model.values():
        for f in s.index:
            if f not in all_features:
                all_features.append(f)
    # Order: spot first, then Kalshi (alphabetical within group)
    spot_first  = [f for f in SPOT_FEATURES   if f in all_features]
    kalshi_next = [f for f in KALSHI_FEATURES if f in all_features]
    order = spot_first + kalshi_next

    fig, ax = plt.subplots(figsize=(22.0, 12.0))
    n_models = len(coefs_by_model)
    width = 0.8 / n_models
    colors = {"Spot-only": "#1f4e8c", "Kalshi-only": "#c1432e", "Combined": "#3a9b6a"}

    for i, (name, s) in enumerate(coefs_by_model.items()):
        vals = [abs(s.get(f, 0.0)) for f in order]
        x = np.arange(len(order)) + (i - (n_models - 1) / 2) * width
        ax.bar(x, vals, width=width, color=colors.get(name, None),
               alpha=0.85, label=name, edgecolor="white", linewidth=0.6)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f.replace("_btc", "").replace("_", " ") for f in order],
                       rotation=35, ha="right")
    ax.set_ylabel("|standardized coefficient|")
    ax.set_title(
        "Feature importance:  standardized |coef| per LR\n"
        "Spot features (left) dominate the Combined model; Kalshi-specific features (right) carry little weight",
        pad=18,
    )
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(0.5, -0.02,
             "Standardized |coefficient| per feature across the three nested LRs.  "
             "Spot features dominate the Combined model; Kalshi-specific features carry little weight.",
             ha="center", va="top", fontsize=22, style="italic", color="#444")

    plt.tight_layout()
    out = FIG_DIR / "feature_importance.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Figure 2 — 2-D decision boundary of the Combined LR
# ---------------------------------------------------------------------------

def plot_decision_boundary(
    df: pd.DataFrame,
    features: list[str],
    combined_pipe,
    x_feat: str = "spot_ret_5m_pre",
    y_feat: str = "opening_pup_btc",
) -> None:
    medians = df[features].median()

    x_lo, x_hi = df[x_feat].quantile([0.02, 0.98])
    y_lo, y_hi = df[y_feat].quantile([0.02, 0.98])
    xs = np.linspace(x_lo, x_hi, 220)
    ys = np.linspace(y_lo, y_hi, 220)
    XX, YY = np.meshgrid(xs, ys)
    grid = pd.DataFrame({f: np.full(XX.size, medians[f]) for f in features})
    grid[x_feat] = XX.ravel()
    grid[y_feat] = YY.ravel()
    prob = combined_pipe.predict_proba(grid[features].values)[:, 1].reshape(XX.shape)

    fig, ax = plt.subplots(figsize=(18.0, 12.0))
    cf = ax.contourf(XX, YY, prob, levels=np.linspace(0.0, 1.0, 21),
                     cmap="RdBu", alpha=0.85)
    ax.contour(XX, YY, prob, levels=[0.5], colors="black", linewidths=2.4)

    plot_df = df.dropna(subset=[x_feat, y_feat, TARGET])
    ups   = plot_df[plot_df[TARGET] == 1]
    downs = plot_df[plot_df[TARGET] == 0]
    ax.scatter(ups[x_feat],   ups[y_feat],   s=55, color="#1f4e8c", alpha=0.55,
               edgecolor="white", linewidth=0.6, label="Resolved UP")
    ax.scatter(downs[x_feat], downs[y_feat], s=55, color="#c1432e", alpha=0.55,
               edgecolor="white", linewidth=0.6, label="Resolved DOWN")

    ax.set_xlabel("Spot 5-min log-return  (pre-contract)")
    ax.set_ylabel("Kalshi opening P(UP)")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(
        "Combined LR decision boundary\n"
        "Background shading = predicted P(UP);  black line = 0.5 threshold",
        pad=18,
    )
    cbar = plt.colorbar(cf, ax=ax, shrink=0.9)
    cbar.set_label("Predicted P(UP)")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    fig.text(0.5, -0.02,
             "Combined LR predicted P(UP) projected onto spot 5-min log-return × Kalshi opening P(UP).  "
             "Other features held at sample medians.",
             ha="center", va="top", fontsize=22, style="italic", color="#444")

    plt.tight_layout()
    out = FIG_DIR / "decision_boundary.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading merged contracts + spot data...")
    merged = pd.read_csv(MERGED, parse_dates=["open_key"])
    merged = merged.dropna(subset=[TARGET]).reset_index(drop=True)
    merged[TARGET] = merged[TARGET].astype(int)

    spot = pd.read_csv(SPOT_CSV, parse_dates=["open_time_utc"])

    print("Building spot features at contract level...")
    df = build_spot_features(merged, spot)
    df = df.dropna(subset=KALSHI_FEATURES + SPOT_FEATURES + [TARGET]).reset_index(drop=True)
    print(f"  Usable rows: {len(df):,}")

    print("Fitting three nested LRs (Spot-only / Kalshi-only / Combined)...")
    coefs_by_model = {
        "Spot-only":   fit_and_get_coefs(df, SPOT_FEATURES),
        "Kalshi-only": fit_and_get_coefs(df, KALSHI_FEATURES),
        "Combined":    fit_and_get_coefs(df, SPOT_FEATURES + KALSHI_FEATURES),
    }

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_feature_importance(coefs_by_model)

    # Combined pipeline (refit) for the decision boundary figure
    combined_features = SPOT_FEATURES + KALSHI_FEATURES
    combined_pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0, random_state=42),
    )
    combined_pipe.fit(df[combined_features].values, df[TARGET].values)
    plot_decision_boundary(df, combined_features, combined_pipe)


if __name__ == "__main__":
    main()
