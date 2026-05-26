import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("kalshi_btc_prices.csv")
df["Correct_bin"] = (df["Correct?"] == "Correct").astype(int)
df["Minute_offset"] = df.groupby("Market Ticker").cumcount()
df["P_majority"] = df.apply(
    lambda r: r["P(UP)"] if r["Prediction"] == "UP" else r["P(DOWN)"], axis=1
)
df["Brier_sq"] = (df["P_majority"] - df["Correct_bin"]) ** 2

brier = df.groupby("Minute_offset")["Brier_sq"].agg(mean_val="mean", std_val="std").reset_index()
x = brier["Minute_offset"].values
y = brier["mean_val"].values
se = brier["std_val"].values / np.sqrt(df.groupby("Minute_offset").size().values)

improvement = (y[0] - y[-1]) / y[0] * 100
half_life = next((i for i, v in enumerate(y) if v < y[0] / 2), None)
n_windows = df["Market Ticker"].nunique()

fig, ax = plt.subplots(figsize=(10, 5))

ax.fill_between(x, y - se, y + se, alpha=0.15, color="black")
ax.plot(x, y, color="black", linewidth=1.8, zorder=3)
ax.scatter(x, y, color="black", s=30, zorder=4)
ax.axhline(0.25, color="gray", linestyle="--", linewidth=1.0, label="Baseline (0.25)")

ax.set_xlim(-0.5, 14.5)
ax.set_ylim(-0.01, 0.30)
ax.set_xticks(range(0, 15))
ax.set_xlabel("Minutes into market window")
ax.set_ylabel("Brier Score")
ax.set_title(f"BTC Brier Score Over Time  (n={n_windows}, {improvement:.1f}% improvement, halves at t={half_life})")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

plt.tight_layout()
plt.savefig("btc_brier_score_timeseries.png", dpi=150, bbox_inches="tight")
print("Saved: btc_brier_score_timeseries.png")
