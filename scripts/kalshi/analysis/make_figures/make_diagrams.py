"""
Section-roadmap diagram for slide 10 of the presentation.

Renders Kalshi P(UP) + Coinbase spot → three diagnostics → information
outcome, with a transparent background so it composes over the deck's
dark-navy chrome. All text in Times New Roman, ≥ 32 pt.

Output: figures/eda/methods_diagram.png
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[4]
FIGS = ROOT / "figures" / "eda"

# Color palette for use over the deck's dark-navy background
NAVY    = "#1f4e8c"
ACCENT  = "#2e7dba"
GREEN   = "#3a9b6a"
AMBER   = "#d49a3a"
SLATE   = "#5a6b80"
LIGHT   = "#e8eef5"
WHITE   = "#ffffff"
YELLOW  = "#f5c518"   # accent for diagnostic boxes on dark bg

mpl.rcParams["font.family"]      = "Times New Roman"
mpl.rcParams["font.serif"]       = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "stix"


def add_box(ax, x, y, w, h, label, sublabel=None,
            face="none", edge=WHITE, text_color=WHITE, fontsize=38,
            sub_fontsize=32, sub_color=WHITE, weight="bold"):
    """Draw a rounded rectangle with a label and optional sublabel below it."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=2.4, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(box)
    cx, cy = x + w / 2, y + h / 2
    if sublabel:
        ax.text(cx, cy + 0.22, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, weight=weight,
                family="Times New Roman")
        ax.text(cx, cy - 0.32, sublabel, ha="center", va="center",
                fontsize=sub_fontsize, color=sub_color, style="italic",
                family="Times New Roman")
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, weight=weight,
                family="Times New Roman")


def add_arrow(ax, x0, y0, x1, y1, color=WHITE, lw=2):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=20,
        color=color, linewidth=lw,
    ))


# ---------------------------------------------------------------------------
# Methods diagram (slide 5)
# ---------------------------------------------------------------------------

def make_methods_diagram():
    """
    Section roadmap shown over the deck's dark-navy background on slide 9.
    All text in Times New Roman at >= 30 pt for 60-person-room legibility.
    """
    fig, ax = plt.subplots(figsize=(22.0, 11.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    # Transparent canvas (rendered over the deck's navy background)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # Column 1: sources (white outline, transparent fill, white text)
    add_box(ax, 0.2, 3.7, 3.2, 1.6, "Kalshi P(UP)",
            sublabel="2,264 BTC contracts",
            face="none", edge=WHITE, text_color=WHITE, sub_color=WHITE,
            fontsize=34, sub_fontsize=30)
    add_box(ax, 0.2, 0.8, 3.2, 1.6, "Coinbase spot",
            sublabel="58,948 1-min bars",
            face="none", edge=WHITE, text_color=WHITE, sub_color=WHITE,
            fontsize=34, sub_fontsize=30)

    # Column 2: diagnostics (yellow outline, transparent fill, yellow text)
    diag_x = 4.6
    diag_w = 5.8
    diag_h = 1.45
    diag_ys = [4.20, 2.30, 0.40]
    diag_specs = [
        ("Lead-lag correlogram",
         r"corr($\Delta$P(UP)$_t$, r$_{t+k}$),  k $\in$ [$-$10, +10]",
         "Cameron, Gelbach & Miller (2008)"),
        ("Variance ratio",
         "Lo & MacKinlay VR(q),  Hurst H",
         "Lo & MacKinlay (1988)"),
        ("Incremental information",
         "Spot vs Kalshi vs combined,  5-fold CV",
         "DeLong et al. (1988)"),
    ]
    for y, (label, sub, cite) in zip(diag_ys, diag_specs):
        add_box(ax, diag_x, y, diag_w, diag_h, label, sublabel=sub,
                face="none", edge=YELLOW, text_color=YELLOW,
                sub_color=YELLOW, fontsize=36, sub_fontsize=32)
        # Citation directly under the box  -- ≥ 32 pt for 60-person-room legibility
        ax.text(diag_x + diag_w / 2, y - 0.32, cite,
                ha="center", va="center",
                fontsize=32, color=WHITE, style="italic",
                family="Times New Roman")

    # Column 3: outcome (yellow outline, transparent fill, yellow text)
    add_box(ax, 10.8, 2.3, 3.0, 1.45, "Information",
            sublabel="between venues",
            face="none", edge=YELLOW, text_color=YELLOW, sub_color=YELLOW,
            fontsize=34, sub_fontsize=30)

    # Arrows: sources to each diagnostic
    for y in diag_ys:
        cy = y + diag_h / 2
        add_arrow(ax, 3.45, 4.4, diag_x - 0.05, cy, color=WHITE, lw=2.0)
        add_arrow(ax, 3.45, 1.6, diag_x - 0.05, cy, color=WHITE, lw=2.0)

    # Arrows: each diagnostic to outcome
    for y in diag_ys:
        cy = y + diag_h / 2
        add_arrow(ax, diag_x + diag_w + 0.05, cy, 10.75, 3.0,
                  color=WHITE, lw=2.0)

    plt.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "methods_diagram.png"
    plt.savefig(out, dpi=220, bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    make_methods_diagram()
