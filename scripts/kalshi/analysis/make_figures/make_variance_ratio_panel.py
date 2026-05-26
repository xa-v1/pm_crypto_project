"""
Generate the variance ratio figure used on slide 11 of the presentation.

Output: figures/varRatio/variance_ratio_panel.png

Left panel:  VR(q) vs q with 95% CIs and the random-walk null at VR=1.
Right panel: log VR(q) vs log q with linear fit (slope = 2H − 1).

All text is set in Times New Roman at >= 30 pt for 60-person-room legibility.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).resolve().parents[4]
VR_JSON = ROOT / "data" / "cleaned" / "variance_ratio.json"
OUT_PNG = ROOT / "figures" / "varRatio" / "variance_ratio_panel.png"

PM_COLOR   = "#c1432e"
SPOT_COLOR = "#1f4e8c"
NULL_COLOR = "#666666"

mpl.rcParams.update({
    "font.family":      "Times New Roman",
    "font.serif":       ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.titlesize":   38,
    "axes.labelsize":   36,
    "xtick.labelsize":  34,
    "ytick.labelsize":  34,
    "legend.fontsize":  34,
})


def main():
    with open(VR_JSON) as f:
        data = json.load(f)

    btc_pm   = data["PM:  P(UP) BTC"]
    btc_spot = data["Spot:  log(close) BTC"]

    qs_str = ["2", "3", "5", "7", "10"]
    qs = np.array([int(q) for q in qs_str])

    pm_vr  = np.array([btc_pm["vr"][q] for q in qs_str])
    pm_lo  = np.array([btc_pm["vr_ci"][q][0] for q in qs_str])
    pm_hi  = np.array([btc_pm["vr_ci"][q][1] for q in qs_str])
    sp_vr  = np.array([btc_spot["vr"][q] for q in qs_str])
    sp_lo  = np.array([btc_spot["vr_ci"][q][0] for q in qs_str])
    sp_hi  = np.array([btc_spot["vr_ci"][q][1] for q in qs_str])

    pm_hurst = btc_pm["hurst"]
    sp_hurst = btc_spot["hurst"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    # --- Left: VR(q) -------------------------------------------------------
    ax1.axhline(1.0, color=NULL_COLOR, linewidth=2.4, linestyle="--",
                label="Random-walk null  VR(q) = 1")
    ax1.errorbar(qs, pm_vr, yerr=[pm_vr - pm_lo, pm_hi - pm_vr],
                 fmt="o", color=PM_COLOR, capsize=10, markersize=20,
                 linewidth=3.5, label="Kalshi P(UP)")
    ax1.errorbar(qs, sp_vr, yerr=[sp_vr - sp_lo, sp_hi - sp_vr],
                 fmt="s", color=SPOT_COLOR, capsize=10, markersize=18,
                 linewidth=3.5, label="BTC spot")

    ax1.set_xlabel("Horizon q (minutes)")
    ax1.set_ylabel("Variance ratio  VR(q)")
    ax1.set_title("Variance ratios, 95% CIs", loc="left", pad=14)
    ax1.set_xticks(qs)
    ax1.set_ylim(0.90, 1.50)
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False)

    # --- Right: log-log with Hurst ----------------------------------------
    log_q  = np.log(qs)
    log_pm = np.log(pm_vr)
    log_sp = np.log(sp_vr)

    pm_slope, pm_int = np.polyfit(log_q, log_pm, 1)
    sp_slope, sp_int = np.polyfit(log_q, log_sp, 1)

    qx = np.linspace(qs.min(), qs.max(), 100)
    lq = np.log(qx)

    ax2.scatter(log_q, log_pm, color=PM_COLOR, s=260, zorder=3)
    ax2.plot(lq, pm_slope * lq + pm_int, color=PM_COLOR, linewidth=3.5,
             label=f"Kalshi P(UP):  H = {pm_hurst:.3f}")
    ax2.scatter(log_q, log_sp, color=SPOT_COLOR, s=230, marker="s", zorder=3)
    ax2.plot(lq, sp_slope * lq + sp_int, color=SPOT_COLOR, linewidth=3.5,
             label=f"BTC spot:  H = {sp_hurst:.3f}")
    ax2.axhline(0.0, color=NULL_COLOR, linewidth=1.8, linestyle=":")

    ax2.set_xlabel("log q")
    ax2.set_ylabel("log VR(q)")
    ax2.set_title("Hurst:  slope = 2H − 1", loc="left", pad=14)
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False)

    fig.text(0.5, -0.01,
             "Lo & MacKinlay VR(q) and Hurst H for Kalshi P(UP) and BTC spot.  "
             "VR(q) = 1 under the random-walk null;  H = 0.5 is the random-walk reference.",
             ha="center", va="top", fontsize=22, style="italic", color="#444")

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {OUT_PNG}")


if __name__ == "__main__":
    main()
