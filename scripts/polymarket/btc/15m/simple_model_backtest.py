import csv
from collections import defaultdict
from datetime import datetime, timedelta

filename = "btc_prices.csv"

# Read all data
with open(filename, "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Organize data by market
markets = defaultdict(list)
for row in rows:
    markets[row["Market Slug"]].append(row)

# Sort each market by timestamp
for slug in markets:
    markets[slug].sort(key=lambda x: datetime.strptime(x["Timestamp"], "%Y-%m-%d %H:%M:%S"))

# Trading results
trades = []
total_profit = 0
total_trades = 0
winning_trades = 0
losing_trades = 0

# Process each market
for slug, market_data in markets.items():
    if len(market_data) != 15:
        continue  # Skip incomplete markets
    
    # Check each minute in the market
    for i, row in enumerate(market_data):
        timestamp = datetime.strptime(row["Timestamp"], "%Y-%m-%d %H:%M:%S")
        
        # Calculate time remaining (15 total minutes, we're at minute i)
        time_remaining = 15 - i
        
        # Only consider entries with <= 7 minutes remaining
        if time_remaining > 7:
            continue
        
        # Skip if we can't calculate 2-minute momentum (need at least 2 prior data points)
        if i < 2:
            continue
        
        # Get probabilities
        p_up_current = float(row["P(UP)"])
        p_down_current = float(row["P(DOWN)"])
        
        # Get probabilities from 2 minutes ago
        row_2min_ago = market_data[i - 2]
        p_up_2min_ago = float(row_2min_ago["P(UP)"])
        p_down_2min_ago = float(row_2min_ago["P(DOWN)"])
        
        # Calculate momentum for both directions
        momentum_up = p_up_current - p_up_2min_ago
        momentum_down = p_down_current - p_down_2min_ago
        
        # Get actual outcome
        actual_outcome = row["Actual Outcome"]
        
        # Check UP bet conditions
        if p_up_current >= 0.75 and momentum_up > 0.05:
            bet_direction = "UP"
            bet_probability = p_up_current
            bet_momentum = momentum_up
            
            # Calculate profit (betting $1)
            # If we bet $1 at probability p, we get $1/p if we win
            potential_payout = 1.0 / p_up_current
            
            if actual_outcome == "Up":
                profit = potential_payout - 1.0  # Net profit (subtract initial $1 bet)
                winning_trades += 1
                win = True
            else:
                profit = -1.0  # Lost our $1 bet
                losing_trades += 1
                win = True
            
            total_profit += profit
            total_trades += 1
            
            trades.append({
                "Market": slug,
                "Timestamp": row["Timestamp"],
                "Time Remaining": time_remaining,
                "Direction": bet_direction,
                "Probability": bet_probability,
                "Momentum": bet_momentum,
                "Actual Outcome": actual_outcome,
                "Win": win,
                "Profit": profit
            })
        
        # Check DOWN bet conditions
        elif p_down_current >= 0.75 and momentum_down > 0.05:
            bet_direction = "DOWN"
            bet_probability = p_down_current
            bet_momentum = momentum_down
            
            # Calculate profit (betting $1)
            potential_payout = 1.0 / p_down_current
            
            if actual_outcome == "Down":
                profit = potential_payout - 1.0
                winning_trades += 1
                win = True
            else:
                profit = -1.0
                losing_trades += 1
                win = False
            
            total_profit += profit
            total_trades += 1
            
            trades.append({
                "Market": slug,
                "Timestamp": row["Timestamp"],
                "Time Remaining": time_remaining,
                "Direction": bet_direction,
                "Probability": bet_probability,
                "Momentum": bet_momentum,
                "Actual Outcome": actual_outcome,
                "Win": win,
                "Profit": profit
            })

# Print results
print("=" * 80)
print("TRADING STRATEGY BACKTEST RESULTS")
print("=" * 80)
print(f"\nStrategy: Enter when time_remaining <= 7min AND P >= 0.75 AND momentum_2min > 0.05")
print(f"\nTotal Trades: {total_trades}")
print(f"Winning Trades: {winning_trades}")
print(f"Losing Trades: {losing_trades}")

if total_trades > 0:
    win_rate = (winning_trades / total_trades) * 100
    avg_profit_per_trade = total_profit / total_trades
    
    print(f"\nWin Rate: {win_rate:.2f}%")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Average Profit per Trade: ${avg_profit_per_trade:.4f}")
    print(f"ROI: {(total_profit / total_trades) * 100:.2f}%")
    
    # Calculate some additional stats
    winning_profits = [t["Profit"] for t in trades if t["Win"]]
    losing_profits = [t["Profit"] for t in trades if not t["Win"]]
    
    if winning_profits:
        avg_win = sum(winning_profits) / len(winning_profits)
        print(f"\nAverage Win: ${avg_win:.4f}")
    
    if losing_profits:
        avg_loss = sum(losing_profits) / len(losing_profits)
        print(f"Average Loss: ${avg_loss:.4f}")
else:
    print("\nNo trades met the criteria!")

print("\n" + "=" * 80)
print("SAMPLE TRADES (First 10)")
print("=" * 80)

# Show first 10 trades
for i, trade in enumerate(trades[:10]):
    print(f"\nTrade #{i+1}:")
    print(f"  Time: {trade['Timestamp']} ({trade['Time Remaining']} min remaining)")
    print(f"  Bet: {trade['Direction']} at {trade['Probability']:.2%} (momentum: {trade['Momentum']:+.4f})")
    print(f"  Outcome: {trade['Actual Outcome']} -> {'WIN' if trade['Win'] else 'LOSS'}")
    print(f"  Profit: ${trade['Profit']:+.4f}")

# Save all trades to CSV
output_filename = "backtest_trades.csv"
with open(output_filename, "w", newline="") as f:
    if trades:
        fieldnames = trades[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)
        print(f"\n\nAll trades saved to {output_filename}")