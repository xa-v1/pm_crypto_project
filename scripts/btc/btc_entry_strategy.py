import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

THRESHOLD   = 0.60
ROLL_WIN    = 3
STAY_CONSEC = 2

df = pd.read_csv("kalshi_btc_prices.csv")
df["Timestamp"]     = pd.to_datetime(df["Timestamp"])
df["Correct_bin"]   = (df["Correct?"] == "Correct").astype(int)
df["Minute_offset"] = df.groupby("Market Ticker").cumcount()

open_p = (
    df[df["Minute_offset"] == 0]
    .set_index("Market Ticker")["P(UP)"]
    .rename("Open_PUP")
)
df = df.join(open_p, on="Market Ticker")

bin_edges  = [0.0, 0.35, 0.45, 0.55, 0.65, 0.75, 1.01]
bin_labels = ["<=0.35", "0.35-0.45", "0.45-0.55", "0.55-0.65", "0.65-0.75", ">=0.75"]
bin_order  = bin_labels[::-1]

df["Open_bin"] = pd.cut(df["Open_PUP"], bins=bin_edges, labels=bin_labels, right=False)

raw = (
    df.groupby(["Open_bin", "Minute_offset"], observed=True)["Correct_bin"]
    .mean()
    .unstack("Minute_offset")
    .reindex(bin_order)
)
n_per_bin = (
    df[df["Minute_offset"] == 0]
    .groupby("Open_bin", observed=True)
    .size()
    .reindex(bin_order)
)
rolled = raw.T.rolling(window=ROLL_WIN, center=True, min_periods=2).mean().T


def first_sustained_cross(series, threshold, consec):
    vals = series.values
    idxs = series.index.tolist()
    for i in range(len(vals) - consec + 1):
        window = vals[i : i + consec]
        if not any(np.isnan(v) for v in window) and all(v >= threshold for v in window):
            return idxs[i]
    return None


rows = []
for b in bin_order:
    em           = first_sustained_cross(rolled.loc[b].dropna(), THRESHOLD, STAY_CONSEC)
    raw_at_em    = raw.loc[b, em]    if em is not None else np.nan
    rolled_at_em = rolled.loc[b, em] if em is not None else np.nan
    peak_rolled  = rolled.loc[b].max()
    peak_minute  = rolled.loc[b].idxmax() if not np.isnan(peak_rolled) else None
    rows.append({
        "P(UP) Bin":             b,
        "n":                     int(n_per_bin[b]),
        "Entry Min":             em if em is not None else "never",
        "Raw Accuracy":               f"{raw_at_em:.1%}"    if not np.isnan(raw_at_em)    else "—",
        "Smoothed Accuracy":          f"{rolled_at_em:.1%}" if not np.isnan(rolled_at_em) else "—",
    })

rule_df = pd.DataFrame(rows)

cols = list(rule_df.columns)
cell_text = rule_df.values.tolist()

entry_col_idx = cols.index("Entry Min")
cmap = plt.get_cmap("RdYlGn")

cell_colours = [["#f5f5f5" if r % 2 == 0 else "white"] * len(cols)
                for r in range(len(cell_text))]

entry_vals = [row[entry_col_idx] for row in cell_text]
numeric_entries = [v for v in entry_vals if isinstance(v, (int, float))]
vmin, vmax = (min(numeric_entries), max(numeric_entries)) if numeric_entries else (0, 14)

for r, val in enumerate(entry_vals):
    if isinstance(val, (int, float)):
        norm = 1.0 - (val - vmin) / max(vmax - vmin, 1)
        rgba = cmap(0.2 + norm * 0.6)
        cell_colours[r][entry_col_idx] = mcolors.to_hex(rgba)
    else:
        cell_colours[r][entry_col_idx] = "#ffcccc"

fig, ax = plt.subplots(figsize=(13, 3.2))
ax.axis("off")

tbl = ax.table(
    cellText=cell_text,
    colLabels=cols,
    cellColours=cell_colours,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.6)

for c in range(len(cols)):
    tbl[0, c].set_facecolor("#2c3e50")
    tbl[0, c].set_text_props(color="white", fontweight="bold")

plt.suptitle(
    f"BTC Conviction-conditional Entry Rule \n"
    f"(Threshold={THRESHOLD:.0%}, Rolling Window={ROLL_WIN}, Stay Consecutively={STAY_CONSEC})",
    fontsize=11, fontweight="bold", y=0.98,
)
plt.tight_layout()
plt.savefig("btc_entry_table.png", dpi=150, bbox_inches="tight")
print("Saved btc_entry_table.png")
plt.show()
