import csv
from collections import Counter

filename = "btc_prices.csv"

with open(filename, "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

# Count rows per market slug
slug_counts = Counter(row["Market Slug"] for row in rows)

# Check if market has all outcomes reported
def market_has_complete_outcomes(market_slug):
    market_rows = [row for row in rows if row["Market Slug"] == market_slug]
    # Check if all rows have non-empty "Actual Outcome"
    return all(row["Actual Outcome"].strip() for row in market_rows)

# Keep only markets with exactly 15 rows AND complete outcomes
filtered_rows = [
    row for row in rows
    if slug_counts[row["Market Slug"]] == 15
    and market_has_complete_outcomes(row["Market Slug"])
]

with open(filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(filtered_rows)

print("Incomplete markets and markets without outcomes removed.")