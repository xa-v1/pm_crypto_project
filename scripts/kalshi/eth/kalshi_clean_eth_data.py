import csv
from collections import Counter

filename = "kalshi_eth_prices.csv"

with open(filename, "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

slug_counts = Counter(row["Market Ticker"] for row in rows)

def market_has_complete_outcomes(market_slug):
    market_rows = [row for row in rows if row["Market Ticker"] == market_slug]
    return all(row["Actual Outcome"].strip() for row in market_rows)

filtered_rows = [
    row for row in rows
    if slug_counts[row["Market Ticker"]] == 15
    and market_has_complete_outcomes(row["Market Ticker"])
]

with open(filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(filtered_rows)

print("Incomplete markets and markets without outcomes removed.")