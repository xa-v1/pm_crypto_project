"""
Appendix figure for the incremental-information LR (slide 13 deep dive).

Fits the three nested logistic regressions used in incremental_information.py
(Spot-only / Kalshi-only / Combined) on the full panel, then produces:

  - figures/logReg/feature_importance.png
        Standardized |coefficient| per feature in each LR, grouped bars.
        Highlights which features carry the directional signal in each model
        and why the Combined model is dominated by spot features.

Reads : data/cleaned/merged_contracts.csv + data/btc/spot_btc_1m.csv
Output: figures/logReg/feature_importance.png
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analysis.logReg.incremental_information import (
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

ROOT       = Path(__file__).resolve().parents[3]
MERGED     = ROOT / "data" / "cleaned" / "merged_contracts.csv"
SPOT_CSV   = ROOT / "data" / "btc"     / "spot_btc_1m.csv"
FIG_DIR    = ROOT / "figures" / "logReg"
TARGET     = "target_btc"

KALSHI_FEATURES = list(_KALSHI_FEATURES_FROM_TEST)
SPOT_FEATURES   = list(_SPOT_FEATURES_FROM_TEST)


def fit_and_get_coefs(df: pd.DataFrame, features: list[str]) -> pd.Series:
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0, random_state=42),
    )
    pipe.fit(df[features].values, df[TARGET].values)
    coef = pipe.named_steps["logisticregression"].coef_[0]
    return pd.Series(coef, index=features)


def plot_feature_importance(coefs_by_model: dict[str, pd.Series]) -> None:
    all_features = []
    for s in coefs_by_model.values():
        for f in s.index:
            if f not in all_features:
                all_features.append(f)
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


if __name__ == "__main__":
    main()
