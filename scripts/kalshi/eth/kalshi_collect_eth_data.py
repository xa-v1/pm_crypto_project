import requests
import json
import time
import csv
from datetime import datetime, timezone
import os

csv_filename = 'kalshi_eth_prices.csv'
BASE_URL = 'https://api.elections.kalshi.com/trade-api/v2'
SERIES_TICKER = 'KXETH15M'

# Set default timeout for all requests
REQUEST_TIMEOUT = 5  # seconds

file_exists = os.path.isfile(csv_filename)

if not file_exists:
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'P(UP)', 'P(DOWN)', 'Momentum', 'Prediction',
                        'Actual Outcome', 'Correct?', 'Market Ticker', 'Volume'])
    print(f"Created new file: {csv_filename}")
else:
    print(f"File exists. Appending to: {csv_filename}")

probabilities_up = []
probabilities_down = []
current_ticker = None
pending_resolutions = []
next_log_time = None
data_points_this_market = 0
market_open_time = None
last_logged_p_up = None

print(f"Tracking ETH 15m markets (Kalshi). Data saving to {csv_filename}")

def safe_request(url, description="request", params=None):
    """Make HTTP request with timeout and error handling"""
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
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

def parse_open_time(open_time_str):
    """Parse ISO 8601 UTC timestamp to Unix timestamp"""
    try:
        dt = datetime.fromisoformat(open_time_str.replace('Z', '+00:00'))
        return int(dt.timestamp())
    except Exception:
        return None

try:
    while True:
        now = int(time.time())

        # 1. DETECT MARKET CHANGE - fetch currently open market from API
        market_data = safe_request(
            f"{BASE_URL}/markets",
            "open market lookup",
            params={'series_ticker': SERIES_TICKER, 'status': 'open', 'limit': 1}
        )

        active_market = None
        if market_data and market_data.get('markets'):
            active_market = market_data['markets'][0]
            ticker = active_market['ticker']

            if current_ticker is not None and ticker != current_ticker:
                print(f"\n--- Market Period Changed ---")
                print(f"Closing: {current_ticker} | Opening: {ticker}")
                print(f"Data points collected for previous market: {data_points_this_market}")
                pending_resolutions.append(current_ticker)
                probabilities_up = []
                probabilities_down = []
                next_log_time = None
                data_points_this_market = 0
                market_open_time = None
                last_logged_p_up = None

            current_ticker = ticker

            # Determine market open time from API open_time field
            if market_open_time is None:
                open_time_str = active_market.get('open_time', '')
                market_open_time = parse_open_time(open_time_str) if open_time_str else now

            if next_log_time is None:
                if now >= market_open_time:
                    next_log_time = now
                else:
                    next_log_time = market_open_time
        else:
            print(f"No open {SERIES_TICKER} market found, waiting...")

        # 2. CHECK PENDING RESOLUTIONS (Non-blocking)
        for p_ticker in pending_resolutions[:]:
            try:
                res_data = safe_request(
                    f"{BASE_URL}/markets/{p_ticker}",
                    f"resolution check for {p_ticker}"
                )

                if res_data and res_data.get('market'):
                    market = res_data['market']
                    result = market.get('result', '')
                    status = market.get('status', '')
                    volume = market.get('volume', 0)

                    # result: 'yes' = BTC went UP, 'no' = BTC went DOWN
                    if result in ('yes', 'no'):
                        outcome = 'UP' if result == 'yes' else 'DOWN'
                        print(f"SUCCESS: Market {p_ticker} resolved to {outcome} | Volume: {volume:,}")

                        try:
                            with open(csv_filename, 'r', newline='') as f:
                                rows = list(csv.reader(f))

                            for i in range(len(rows)):
                                if len(rows[i]) > 7 and rows[i][7] == p_ticker:
                                    while len(rows[i]) < 9:
                                        rows[i].append('')
                                        rows[i][5] = outcome
                                        rows[i][6] = 'Correct' if rows[i][4].upper() == outcome.upper() else 'Incorrect'
                                        rows[i][8] = str(volume)

                            with open(csv_filename, 'w', newline='') as f:
                                csv.writer(f).writerows(rows)

                            pending_resolutions.remove(p_ticker)
                        except Exception as csv_error:
                            print(f"Error updating CSV for {p_ticker}: {csv_error}")
            except Exception as e:
                print(f"Error checking resolution for {p_ticker}: {e}")

        # 3. COLLECT LIVE DATA
        if active_market:
            try:
                # yes_bid_dollars / yes_ask_dollars are strings like "0.8100"
                # Fallback to yes_bid / yes_ask (integer cents) if dollar fields absent
                yes_bid_raw = active_market.get('yes_bid_dollars')
                yes_ask_raw = active_market.get('yes_ask_dollars')

                if yes_bid_raw is not None and yes_ask_raw is not None:
                    yes_bid = float(yes_bid_raw)
                    yes_ask = float(yes_ask_raw)
                else:
                    yes_bid = active_market.get('yes_bid', 0) / 100.0
                    yes_ask = active_market.get('yes_ask', 0) / 100.0

                mid_prob_up = (yes_bid + yes_ask) / 2
                probabilities_up.append(mid_prob_up)
                probabilities_down.append(1 - mid_prob_up)

                # Check if it's time to log (exactly 15 times per market)
                if now >= next_log_time and data_points_this_market < 15:
                    if probabilities_up:
                        avg_up = sum(probabilities_up) / len(probabilities_up)
                        avg_down = sum(probabilities_down) / len(probabilities_down)
                        prediction = 'UP' if avg_up > avg_down else 'DOWN'

                        # For first point, use market open time; otherwise use scheduled time
                        if data_points_this_market == 0:
                            ts = datetime.fromtimestamp(market_open_time).strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            ts = datetime.fromtimestamp(next_log_time).strftime('%Y-%m-%d %H:%M:%S')

                        with open(csv_filename, 'a', newline='') as f:
                            writer = csv.writer(f)
                            momentum = round(avg_up - last_logged_p_up, 4) if last_logged_p_up is not None else 0
                            last_logged_p_up = avg_up
                            writer.writerow([ts, f"{avg_up:.4f}", f"{avg_down:.4f}", momentum, prediction, '', '', current_ticker])

                        print(f"Logged data point {data_points_this_market + 1}/15 at {ts}")
                        data_points_this_market += 1
                        probabilities_up, probabilities_down = [], []

                        # Schedule next log time
                        if data_points_this_market == 1:
                            next_log_time = market_open_time + 60
                        else:
                            next_log_time += 60

            except Exception as e:
                print(f"Live data error: {e}")

        time.sleep(10)

except KeyboardInterrupt:
    print(f"\nStopped. Data saved to {csv_filename}")
