import pandas as pd
import matplotlib.pyplot as plt

OVERCONFIDENCE_THRESHOLD = 0.05  

df = pd.read_csv("kalshi_eth_prices.csv")
df["Timestamp"]     = pd.to_datetime(df["Timestamp"])
df["Correct_bin"]   = (df["Correct?"] == "Correct").astype(int)
df["Minute_offset"] = df.groupby("Market Ticker").cumcount()

open_hour = (
    df[df["Minute_offset"] == 0]
    .set_index("Market Ticker")["Timestamp"]
    .dt.hour
    .rename("Open_hour")
)
df = df.join(open_hour, on="Market Ticker")

opens = df[df["Minute_offset"] == 0].copy()

opens["P_majority"] = opens.apply(
    lambda r: r["P(UP)"] if r["Prediction"] == "UP" else r["P(DOWN)"], axis=1
)

hour_stats = (
    opens.groupby("Open_hour")
    .agg(
        accuracy=("Correct_bin", "mean"),
        implied_acc=("P_majority", "mean"),
        n=("Correct_bin", "count"),
    )
    .reset_index()
)

hour_stats["overconf_gap"] = hour_stats["implied_acc"] - hour_stats["accuracy"]
hour_stats["overconfident"] = hour_stats["overconf_gap"] > OVERCONFIDENCE_THRESHOLD

hours = hour_stats["Open_hour"].values
acc   = hour_stats["accuracy"].values
implied = hour_stats["implied_acc"].values
is_oc = hour_stats["overconfident"].values

flagged = hour_stats[hour_stats["overconfident"]].copy()
if not flagged.empty:
    print(f"\nOverconfident hours  (implied acc − actual acc > {OVERCONFIDENCE_THRESHOLD:.0%})")
    print("=" * 60)
    for _, row in flagged.iterrows():
        print(
            f"  {int(row['Open_hour']):02d}:00  "
            f"implied={row['implied_acc']:.1%}  "
            f"actual={row['accuracy']:.1%}  "
            f"gap={row['overconf_gap']:+.1%}  "
            f"n={int(row['n'])}"
        )
else:
    print("\nNo systematically overconfident hours detected.")

fig, ax = plt.subplots(figsize=(14, 5))

bar_colors = ["#c0392b" if oc else "#2980b9" for oc in is_oc]
ax.bar(hours, acc, color=bar_colors, width=0.7, edgecolor="white", linewidth=0.6,
       zorder=3, label="Actual Accuracy")

ax.plot(hours, implied, color="black", linewidth=1.4, linestyle="--",
        marker="D", markersize=4, zorder=5, label="Implied Accuracy")

for h, ac, im, oc in zip(hours, acc, implied, is_oc):
    if oc:
        ax.annotate(
            f"−{im - ac:.0%}",
            xy=(h, ac),
            xytext=(h, (ac + im) / 2),
            ha="center", va="center", fontsize=7, color="#c0392b",
            fontweight="bold",
        )

ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.9, alpha=0.6,
           label="50% Baseline")
ax.set_xticks(hours)
ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha="right")
ax.set_xlabel("Hour of contract open (UTC)")
ax.set_ylabel("Accuracy")
ax.set_title(
    "Average Initial Prediction Accuracy by Hour  (ETH Kalshi 15-min Contracts)"
)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
ax.set_ylim(bottom=0.25)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("eth_hourly_seasonality.png", dpi=150, bbox_inches="tight")
print("\nSaved: eth_hourly_seasonality.png")
