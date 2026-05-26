import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests
import pandas as pd


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
EVENT_SLUG = "what-price-will-bitcoin-hit-in-february-2026"
FIDELITY = 60

START_TS = int(datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())


def get_end_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _parse_json_field(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _parse_resolution_date(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def fetch_event_markets(slug: str) -> list[dict]:
    url = f"{GAMMA_API}/events"
    resp = requests.get(url, params={"slug": slug}, timeout=15)
    resp.raise_for_status()
    data = resp.json()  # <-- removed the bad debug line that was here

    if not data:
        raise ValueError(f"No event found for slug: {slug!r}")

    event = data[0]
    markets = event.get("markets", [])
    if not markets:
        raise ValueError("Event has no sub-markets.")

    active_questions = {
        m.get("question")
        for m in markets
        if not (bool(m.get("closed")) and m.get("umaResolutionStatus") == "resolved")
        and m.get("clobTokenIds")
    }

    result = []
    for m in markets:
        question = m.get("question", "Unknown")
        token_ids = _parse_json_field(m.get("clobTokenIds", []))

        if not token_ids:
            print(f"  SKIP (no clobTokenIds): {question}")
            continue

        resolved = bool(m.get("closed")) and m.get(
            "umaResolutionStatus") == "resolved"

        if resolved and question in active_questions:
            resolved = False
            resolution_price = None
            resolution_date = None

        resolution_price = None
        resolution_date = None

        if resolved:
            outcome_prices = _parse_json_field(m.get("outcomePrices", "[]"))
            if outcome_prices:
                try:
                    resolution_price = float(outcome_prices[0])
                except (TypeError, ValueError):
                    pass

            raw_closed = m.get("closedTime") or m.get("endDate")
            resolution_date = _parse_resolution_date(
                raw_closed.replace(" ", "T") if raw_closed else None
            )

        result.append({
            "label":            question,
            "yes_token_id":     token_ids[0],
            "resolved":         resolved,
            "resolution_price": resolution_price,
            "resolution_date":  resolution_date,
        })

    return result


MAX_CHUNK_SECONDS = 14 * 24 * 3600


def fetch_daily_avg_prices(token_id: str, start_ts: int, end_ts: int) -> list[dict]:
    url = f"{CLOB_API}/prices-history"
    daily: dict[str, list[float]] = defaultdict(list)

    chunk_start = start_ts
    while chunk_start < end_ts:
        chunk_end = min(chunk_start + MAX_CHUNK_SECONDS, end_ts)
        params = {
            "market":   token_id,
            "startTs":  chunk_start,
            "endTs":    chunk_end,
            "fidelity": FIDELITY,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()

        history = resp.json().get("history", [])

        for point in history:
            ts = int(point["t"])

            # FIX: skip any points before February 1st
            if ts < start_ts:
                continue

            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            day = dt.strftime("%Y-%m-%d")
            daily[day].append(float(point["p"]))

        chunk_start = chunk_end
        if chunk_start < end_ts:
            time.sleep(0.1)

    return [
        {"date": day, "price": round(sum(prices) / len(prices), 6)}
        for day, prices in sorted(daily.items())
    ]


def shorten_label(question: str) -> str:
    q = question.lower()
    for verb in ("reach", "dip to", "dip"):
        if verb in q:
            action = verb.replace(" to", "")
            m = re.search(r"\$([\d,]+)", question)
            if m:
                val = int(m.group(1).replace(",", ""))
                return f"{action} ${val // 1000}k" if val >= 1_000 else f"{action} ${val}"
    return question


def build_dataframe(markets: list[dict], start_ts: int, end_ts: int) -> pd.DataFrame:
    rows = []
    for mkt in markets:
        label = shorten_label(mkt["label"])
        token_id = mkt["yes_token_id"]
        resolved = mkt["resolved"]

        status = f"  [RESOLVED → {mkt['resolution_price']} on {mkt['resolution_date']}]" if resolved else ""
        print(f"  {label:<20}  token …{token_id[-8:]}{status}")

        try:
            points = fetch_daily_avg_prices(token_id, start_ts, end_ts)
            if not points:
                print(f"    WARNING: no price data for {label!r}.")
            for p in points:
                rows.append(
                    {"date": p["date"], "outcome_label": label, "price": p["price"]})
        except requests.HTTPError as exc:
            print(
                f"    WARNING: HTTP {exc.response.status_code} for {label!r}, skipping.")
        except Exception as exc:
            print(
                f"    WARNING: {type(exc).__name__} for {label!r}: {exc}, skipping.")

        if resolved and mkt["resolution_price"] is not None and mkt["resolution_date"]:
            rows.append({
                "date":          f"{mkt['resolution_date']}_resolved",
                "outcome_label": label,
                "price":         mkt["resolution_price"],
            })

        time.sleep(0.25)

    if not rows:
        return pd.DataFrame(columns=["date", "outcome_label", "price"])

    df = pd.DataFrame(rows)

    is_resolved = df["date"].str.endswith("_resolved")
    df_avg = df[~is_resolved].copy()
    df_res = df[is_resolved].copy()

    df_res["date"] = df_res["date"].str.replace("_resolved", "", regex=False)
    df_avg["date"] = pd.to_datetime(df_avg["date"])
    df_res["date"] = pd.to_datetime(df_res["date"])

    df_avg = (
        df_avg.groupby(["date", "outcome_label"], as_index=False)["price"]
        .mean()
        .round({"price": 6})
    )

    df_avg.sort_values(["date", "outcome_label"], inplace=True)
    df_res.sort_values(["date", "outcome_label"], inplace=True)

    df_avg = df_avg[
        ~df_avg.set_index(["date", "outcome_label"]).index.isin(
            df_res.set_index(["date", "outcome_label"]).index
        )
    ]

    df_res["date"] = df_res["date"].dt.strftime("%Y-%m-%d") + "_resolved"
    df_avg["date"] = df_avg["date"].dt.strftime("%Y-%m-%d")
    df = pd.concat([df_avg, df_res], ignore_index=True)

    def label_sort_key(lbl):
        kind = 0 if "dip" in lbl else 1
        num = int("".join(filter(str.isdigit, lbl)) or 0)
        return (kind, num)

    df["_label_key"] = df["outcome_label"].map(label_sort_key)
    df["_date_key"] = df["date"].str.replace("_resolved", "z", regex=False)
    df.sort_values(["_label_key", "_date_key"], inplace=True)
    df.drop(columns=["_label_key", "_date_key"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    end_ts = get_end_ts()
    start_dt = datetime.fromtimestamp(START_TS, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    print(f"Event slug  : {EVENT_SLUG}")
    print(
        f"Range       : {start_dt.date()} 00:00 UTC  →  {end_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print()

    print("Step 1 — Discovering sub-markets from Gamma Events API...")
    markets = fetch_event_markets(EVENT_SLUG)
    print(f"  Found {len(markets)} sub-market(s)")
    print()

    print("Step 2 — Fetching hourly CLOB price history (daily averages)...")
    df = build_dataframe(markets, START_TS, end_ts)
    print()

    if df.empty:
        print("No data collected — nothing to save.")
        return

    out_path = "polymarket_btc_february_prices.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")
    print()

    df_pivot = df[~df["date"].str.endswith("_resolved")].copy()
    df_pivot["date"] = pd.to_datetime(df_pivot["date"])
    print("Summary — Daily average Yes-token price (0=No, 1=Yes) per day:")
    pivot = df_pivot.pivot_table(
        index="date",
        columns="outcome_label",
        values="price",
        aggfunc="first",
    )
    pivot.index = pivot.index.strftime("%Y-%m-%d")
    col_order = sorted(pivot.columns, key=lambda c: (
        0 if "dip" in c else 1,
        int("".join(filter(str.isdigit, c)) or 0),
    ))
    pivot = pivot[col_order]
    print(pivot.to_string())

    df_res = df[df["date"].str.endswith("_resolved")]
    if not df_res.empty:
        print()
        print("Resolved markets:")
        for _, row in df_res.iterrows():
            date = row["date"].replace("_resolved", "")
            label = row["outcome_label"]
            price = row["price"]
            outcome = "YES" if price >= 0.5 else "NO"
            print(f"  {label:<20}  resolved {date}  →  {outcome} ({price})")


if __name__ == "__main__":
    main()
