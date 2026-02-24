import requests
import json
import time
import csv
from datetime import datetime
import os

csv_filename = 'eth_prices.csv'
asset_ticker = 'eth'

REQUEST_TIMEOUT = 5

file_exists = os.path.isfile(csv_filename)

if not file_exists:
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'P(UP)', 'P(DOWN)', 'Prediction',
                        'Actual Outcome', 'Correct?', 'Market Slug', 'Volume'])
    print(f"Created new file: {csv_filename}")
else:
    print(f"File exists. Appending to: {csv_filename}")

probabilities_up = []
probabilities_down = []
current_slug = None
pending_resolutions = []
next_log_time = None
data_points_this_market = 0

print(f"Tracking ETH 15m markets")

def safe_request(url, description="request"):
    """Make HTTP request with timeout and error handling"""
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print(f"Timeout on {description}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error on {description}: {type(e).__name__}")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON response on {description}")
        return None

try:
    while True:
        now = int(time.time())
        period = now - (now % 900)
        slug = f"eth-updown-15m-{period}"
        market_start_time = period

        if current_slug is not None and slug != current_slug:
            print(f"\n--- Market Period Changed ---")
            print(f"Closing: {current_slug} | Opening: {slug}")
            print(f"Data points collected for previous market: {data_points_this_market}")
            pending_resolutions.append(current_slug)
            probabilities_up = []
            probabilities_down = []
            next_log_time = None
            data_points_this_market = 0

        current_slug = slug

        if next_log_time is None:
            if now >= market_start_time:
                next_log_time = now
            else:
                next_log_time = market_start_time

        for p_slug in pending_resolutions[:]:
            try:
                m_data = safe_request(
                    f"https://gamma-api.polymarket.com/markets?slug={p_slug}",
                    f"resolution check for {p_slug}"
                )

                if m_data and len(m_data) > 0:
                    market = m_data[0]
                    if market.get('umaResolutionStatus') == 'resolved' or market.get('closed'):
                        outcomes = json.loads(market.get('outcomes', '[]'))
                        prices = json.loads(market.get('outcomePrices', '[]'))
                        volume = float(market.get('volume', 0))

                        outcome = None
                        for i, p in enumerate(prices):
                            if float(p) >= 0.99:
                                outcome = outcomes[i]
                                break

                        if outcome:
                            print(f"Previous Market ({p_slug}) resolved to {outcome} | Volume: ${volume:,.2f}")

                            try:
                                with open(csv_filename, 'r', newline='') as f:
                                    rows = list(csv.reader(f))

                                for i in range(len(rows)):
                                    if len(rows[i]) > 6 and rows[i][6] == p_slug:
                                        rows[i][4] = outcome
                                        rows[i][5] = 'Correct' if rows[i][3].upper() == outcome.upper() else 'Incorrect'
                                        if len(rows[i]) > 7:
                                            rows[i][7] = f"{volume:.2f}"
                                        else:
                                            rows[i].append(f"{volume:.2f}")

                                with open(csv_filename, 'w', newline='') as f:
                                    csv.writer(f).writerows(rows)

                                pending_resolutions.remove(p_slug)
                            except Exception as csv_error:
                                print(f"Error updating CSV for {p_slug}: {csv_error}")
            except Exception as e:
                print(f"Error checking resolution for {p_slug}: {e}")

        try:
            market_json = safe_request(
                f"https://gamma-api.polymarket.com/markets?slug={slug}",
                f"market data for {slug}"
            )

            if market_json and len(market_json) > 0:
                token_ids = json.loads(market_json[0].get('clobTokenIds', '[]'))
                
                if token_ids:
                    bid_data = safe_request(
                        f"https://clob.polymarket.com/price?token_id={token_ids[0]}&side=buy",
                        "bid price"
                    )
                    ask_data = safe_request(
                        f"https://clob.polymarket.com/price?token_id={token_ids[0]}&side=sell",
                        "ask price"
                    )

                    if bid_data and ask_data:
                        mid_prob = (float(bid_data['price']) + float(ask_data['price'])) / 2
                        probabilities_up.append(mid_prob)
                        probabilities_down.append(1 - mid_prob)

                        if now >= next_log_time and data_points_this_market < 15:
                            if probabilities_up:
                                avg_up = sum(probabilities_up) / len(probabilities_up)
                                avg_down = sum(probabilities_down) / len(probabilities_down)
                                prediction = 'UP' if avg_up > avg_down else 'DOWN'

                                if data_points_this_market == 0:
                                    ts = datetime.fromtimestamp(market_start_time).strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    ts = datetime.fromtimestamp(next_log_time).strftime('%Y-%m-%d %H:%M:%S')

                                with open(csv_filename, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([ts, f"{avg_up:.4f}", f"{avg_down:.4f}", prediction, '', '', slug])

                                print(f"Logged data point {data_points_this_market + 1}/15 at {ts}")
                                data_points_this_market += 1
                                probabilities_up, probabilities_down = [], []

                                if data_points_this_market == 1:
                                    next_log_time = market_start_time + 60
                                else:
                                    next_log_time += 60
            else:
                print(f"Waiting for market {slug} to go live...")

        except Exception as e:
            print(f"Live data error: {e}")

        time.sleep(10)

except KeyboardInterrupt:
    print(f"\nStopped. Data saved to {csv_filename}")