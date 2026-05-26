import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("kalshi_btc_prices.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Correct_bin"] = (df["Correct?"] == "Correct").astype(int)
df["Minute_offset"] = df.groupby("Market Ticker").cumcount()

accuracy = (
    df.groupby("Minute_offset")["Correct_bin"]
    .agg(accuracy="mean", n="count")
    .reset_index()
)
accuracy["accuracy_pct"] = accuracy["accuracy"] * 100

fig, ax = plt.subplots(figsize=(10, 5))

x = accuracy["Minute_offset"].values
y = accuracy["accuracy_pct"].values

ax.plot(x, y, color="black", linewidth=1.8, zorder=3)
ax.scatter(x, y, color="black", s=30, zorder=4)
ax.axhline(50, color="gray", linestyle="--", linewidth=1.0, label="50% (random chance)")
ax.axvline(15, color="gray", linestyle=":", linewidth=1.0, label="Outcome set (t=15)")

total_windows = df["Market Ticker"].nunique()
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(40, 100)
ax.set_xticks(range(0, 16))
ax.set_xlabel("Minutes into market window")
ax.set_ylabel("Majority prediction accuracy (%)")
ax.set_title(f"BTC Majority Prediction Accuracy Over Time  (n={total_windows} windows)")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

plt.tight_layout()
plt.savefig("btc_accuracy_timeseries.png", dpi=150, bbox_inches="tight")
print("Saved: btc_accuracy_timeseries.png")
print(accuracy.to_string(index=False))
