"""
Pull 1-minute spot bars for BTC-USD and ETH-USD from Coinbase Exchange public
API over the Kalshi data window. No auth required.

Coinbase preferred over Binance.US: latter has 45%+ zero-volume bars at
1-minute granularity in this period, which attenuates any short-horizon signal.

Outputs (timestamps stored as UTC ISO strings):
    data/spot_btc_1m.csv
    data/spot_eth_1m.csv

Each row: open_time_utc, open, high, low, close, volume.
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

BASE = "https://api.exchange.coinbase.com/products/{product}/candles"
GRANULARITY = 60          # seconds → 1m candles
MAX_CANDLES = 300         # Coinbase per-request cap
CHUNK_SECONDS = MAX_CANDLES * GRANULARITY  # 18,000 s = 300 min per chunk

PT_START = "2026-02-15 00:00:00"
PT_END   = "2026-03-28 00:00:00"


def pt_to_utc(ts_str: str) -> datetime:
    return pd.Timestamp(ts_str, tz="America/Los_Angeles").tz_convert("UTC").to_pydatetime()


def fetch_candles(product: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    rows = []
    cursor = start_utc
    n_chunks = 0
    while cursor < end_utc:
        chunk_end = min(cursor + timedelta(seconds=CHUNK_SECONDS - GRANULARITY), end_utc)
        r = requests.get(
            BASE.format(product=product),
            params={
                "start": cursor.isoformat(),
                "end":   chunk_end.isoformat(),
                "granularity": GRANULARITY,
            },
            headers={"User-Agent": "research-script"},
            timeout=30,
        )
        if r.status_code == 429:
            time.sleep(1.0)
            continue
        r.raise_for_status()
        batch = r.json()
        # Coinbase returns: [[time, low, high, open, close, volume], ...] newest-first
        if batch:
            rows.extend(batch)
        cursor = chunk_end + timedelta(seconds=GRANULARITY)
        n_chunks += 1
        if n_chunks % 25 == 0:
            print(f"    fetched {n_chunks} chunks, {len(rows):,} bars so far ...")
        time.sleep(0.12)  # public rate limit ~10 req/s; stay well below

    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["open_time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop_duplicates(subset="open_time_utc").sort_values("open_time_utc")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df[["open_time_utc", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def main():
    start_utc = pt_to_utc(PT_START)
    end_utc   = pt_to_utc(PT_END)
    print(f"Fetching {start_utc}  →  {end_utc}  (UTC)")

    for product, fname in [("BTC-USD", "spot_btc_1m.csv"), ("ETH-USD", "spot_eth_1m.csv")]:
        print(f"\n  {product} ...")
        df = fetch_candles(product, start_utc, end_utc)
        out_path = DATA_DIR / fname
        df.to_csv(out_path, index=False)
        zero_vol = (df["volume"] == 0).sum()
        print(f"  {product}  {len(df):,} bars  ({zero_vol:,} zero-volume = {zero_vol/len(df)*100:.1f}%)")
        print(f"           range: {df['open_time_utc'].iloc[0]}  →  {df['open_time_utc'].iloc[-1]}")
        print(f"           saved →  {out_path}")


if __name__ == "__main__":
    main()
