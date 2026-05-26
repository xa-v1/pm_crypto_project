import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

df = pd.read_csv("kalshi_eth_prices.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Correct_bin"] = (df["Correct?"] == "Correct").astype(int)
df["Minute_offset"] = df.groupby("Market Ticker").cumcount()

open_p = (
    df[df["Minute_offset"] == 0]
    .set_index("Market Ticker")["P(UP)"]
    .rename("Open_PUP")
)
df = df.join(open_p, on="Market Ticker")

bin_edges   = [0.0, 0.35, 0.45, 0.55, 0.65, 0.75, 1.01]
bin_labels  = ["<=0.35", "0.35-0.45", "0.45-0.55",
               "0.55-0.65", "0.65-0.75", ">=0.75"]
df["Open_bin"] = pd.cut(
    df["Open_PUP"],
    bins=bin_edges,
    labels=bin_labels,
    right=False,
)

pivot = (
    df.groupby(["Open_bin", "Minute_offset"], observed=True)["Correct_bin"]
    .mean()
    .unstack("Minute_offset")
)
pivot = pivot.iloc[::-1]

n_per_bin = (
    df[df["Minute_offset"] == 0]
    .groupby("Open_bin", observed=True)
    .size()
    .iloc[::-1]
)

fig, ax = plt.subplots(figsize=(13, 5))

im = ax.imshow(
    pivot.values,
    aspect="auto",
    cmap="RdYlGn",
    vmin=0.30,
    vmax=1.0,
    interpolation="nearest",
)

for r, row_label in enumerate(pivot.index):
    for c, col_label in enumerate(pivot.columns):
        val = pivot.values[r, c]
        if not np.isnan(val):
            txt_color = "black" if 0.42 < val < 0.78 else "white"
            ax.text(c, r, f"{val:.0%}", ha="center", va="center",
                    fontsize=7.5, color=txt_color, fontweight="bold")

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_xlabel("Minute Within Contract")

ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(
    [f"{lbl}" for lbl in pivot.index],
    fontsize=9,
)
ax.set_ylabel("P(UP) Bin at Contract Open")

ax.set_title(
    "Kalshi 15-min ETH contracts \n"
    "Prediction Accuracy by (Minute, Opening Conviction)\n",
    fontsize=11,
    pad=10,
)

ax.axvline(0.5, color="white", linewidth=0.6, alpha=0.5)

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Accuracy", fontsize=9)
cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

plt.tight_layout()
plt.savefig("eth_accuracy_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved: eth_accuracy_heatmap.png")

print("\nMean accuracy by bin × minute:")
with pd.option_context("display.float_format", "{:.1%}".format,
                       "display.max_columns", 20):
    print(pivot.to_string())
