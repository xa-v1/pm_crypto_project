"""
BRTI + Kalshi Integrated Paper Trader
======================================

Combines:
  1. BRTI Replicator — live BTC fair price from 4 exchange order books
  2. Kalshi KXBTC15M poller — live binary market probabilities
  3. Paper Trading Engine — simulated trades when edge is detected

Architecture:
  - BRTI websocket feeds run continuously (Coinbase, Kraken, Bitstamp, Gemini)
  - Kalshi API is polled every 10s for the active KXBTC15M market
  - Every second, the system compares BRTI-derived fair P(UP) vs Kalshi P(UP)
  - When edge exceeds threshold, a simulated trade is placed
  - Positions auto-close at market expiry based on BRTI price vs window start

Usage:
  pip install websockets aiohttp numpy requests
  python brti_kalshi_integrated.py
  python brti_kalshi_integrated.py --aggressive   # Lower edge threshold (3%)
  python brti_kalshi_integrated.py --log trades.csv  # Log trades to CSV
"""

from datetime import datetime, timezone
import sys as _sys
import asyncio
import json
import time
import math
import csv
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("brti_kalshi")


# ═════════════════════════════════════════════════════════
# BRTI Configuration (CME CF RTI Methodology v16.4)
# ═════════════════════════════════════════════════════════

LAMBDA = 1.0 / 0.37
SPACING = 1
MID_SPREAD_THRESHOLD = 0.005
STALE_THRESHOLD_SEC = 30
OUTLIER_THRESHOLD = 0.05
CALC_INTERVAL_SEC = 1.0


# ═════════════════════════════════════════════════════════
# Paper Trading Configuration
# ═════════════════════════════════════════════════════════

DEFAULT_EDGE_THRESHOLD = 0.05    # 5% minimum edge to trade
AGGRESSIVE_EDGE_THRESHOLD = 0.03  # 3% for aggressive mode
MAX_POSITION_SIZE = 100          # Max $100 per trade (Kalshi contract = $1)
MIN_POSITION_SIZE = 5            # Min $5 per trade
BANKROLL = 1000.0                # Starting paper bankroll
KELLY_FRACTION = 0.25            # Quarter-Kelly for safety
MAX_OPEN_POSITIONS = 3           # Max simultaneous open positions


# ═════════════════════════════════════════════════════════
# Dynamic Order Size Cap (Section 4.1.3)
# ═════════════════════════════════════════════════════════

def compute_dynamic_order_size_cap(raw_bids, raw_asks,
                                   n_ask=10, n_bid=10, trim_frac=0.25):
    if not raw_asks or not raw_bids:
        return 100.0
    ask_sizes = [s for _, s in raw_asks[:n_ask]]
    bid_sizes = [s for _, s in raw_bids[:n_bid]]
    S = sorted(ask_sizes + bid_sizes)
    n = len(S)
    if n < 4:
        return 100.0
    k = max(1, int(math.floor(trim_frac * n)))
    S_w = list(S)
    for i in range(k):
        S_w[i] = S[k]
    for i in range(n - k, n):
        S_w[i] = S[n - k - 1]
    win_mean = sum(S_w) / n
    win_std = math.sqrt(sum((x - win_mean) ** 2 for x in S_w) / (n - 1))
    return max(win_mean + win_std, 1.0)


# ═════════════════════════════════════════════════════════
# Data Structures
# ═════════════════════════════════════════════════════════

@dataclass
class OrderBook:
    exchange: str
    bids: list = field(default_factory=list)
    asks: list = field(default_factory=list)
    last_update: float = 0.0

    @property
    def mid_price(self) -> Optional[float]:
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2
        return None

    @property
    def is_stale(self):
        return (time.time() - self.last_update) > STALE_THRESHOLD_SEC

    @property
    def crosses(self):
        if self.bids and self.asks:
            return self.bids[0][0] >= self.asks[0][0]
        return False

    @property
    def is_valid(self):
        return bool(self.bids) and bool(self.asks) and not self.is_stale and not self.crosses


@dataclass
class BRTIResult:
    timestamp: float
    value: float
    utilized_depth: int
    num_exchanges: int
    exchange_mids: dict
    spread_bps: float
    order_size_cap: float


@dataclass
class KalshiMarketState:
    """Current state of the active Kalshi KXBTC15M market."""
    ticker: str = ""
    p_up: float = 0.5           # Mid probability that BTC goes up
    p_down: float = 0.5
    yes_bid: float = 0.0
    yes_ask: float = 0.0
    volume: int = 0
    open_time: Optional[float] = None
    close_time: Optional[float] = None
    last_poll: float = 0.0
    is_active: bool = False


@dataclass
class PaperPosition:
    """A single simulated trade."""
    id: int
    side: str                    # "UP" or "DOWN"
    entry_price: float           # Price paid per contract (0-1)
    size: int                    # Number of contracts
    market_ticker: str
    brti_at_entry: float
    window_start_price: float    # BRTI price at market open
    entry_time: float
    edge_at_entry: float
    fair_p_at_entry: float
    kalshi_p_at_entry: float
    # Filled on close
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    outcome: Optional[str] = None
    exit_time: Optional[float] = None
    exit_reason: Optional[str] = None


# ═════════════════════════════════════════════════════════
# BRTI Calculation Engine (unchanged from original)
# ═════════════════════════════════════════════════════════

class BRTIEngine:
    def __init__(self):
        self.order_books: dict[str, OrderBook] = {}
        self.history: list[BRTIResult] = []
        self._flagged_exchanges: dict[str, float] = {}

    def update_order_book(self, exchange, bids, asks):
        clean_bids = sorted([(p, s) for p, s in bids if p >
                            0 and s > 0], key=lambda x: -x[0])
        clean_asks = sorted([(p, s) for p, s in asks if p >
                            0 and s > 0], key=lambda x: x[0])
        self.order_books[exchange] = OrderBook(
            exchange=exchange, bids=clean_bids, asks=clean_asks, last_update=time.time()
        )

    def _filter_exchanges(self):
        valid = {k: v for k, v in self.order_books.items() if v.is_valid}
        if len(valid) < 2:
            return valid
        mids = {k: v.mid_price for k, v in valid.items() if v.mid_price}
        if not mids:
            return {}
        median_mid = float(np.median(list(mids.values())))

        reinstate_threshold = OUTLIER_THRESHOLD * 0.5
        for ex in list(self._flagged_exchanges.keys()):
            if ex in mids:
                if abs(mids[ex] - median_mid) / median_mid < reinstate_threshold:
                    del self._flagged_exchanges[ex]

        filtered = {}
        for k, v in valid.items():
            mp = v.mid_price
            if mp is None:
                continue
            dev = abs(mp - median_mid) / median_mid
            if dev > OUTLIER_THRESHOLD or k in self._flagged_exchanges:
                if k not in self._flagged_exchanges:
                    self._flagged_exchanges[k] = time.time()
            else:
                filtered[k] = v
        return filtered

    def _apply_cap_and_consolidate(self, books):
        raw_bids, raw_asks = [], []
        for ob in books.values():
            raw_bids.extend(ob.bids)
            raw_asks.extend(ob.asks)
        raw_bids.sort(key=lambda x: -x[0])
        raw_asks.sort(key=lambda x: x[0])
        if not raw_bids or not raw_asks:
            return [], [], 100.0
        cap = compute_dynamic_order_size_cap(raw_bids, raw_asks)
        capped_bids = [(p, min(s, cap)) for p, s in raw_bids]
        capped_asks = [(p, min(s, cap)) for p, s in raw_asks]
        return capped_bids, capped_asks, cap

    def _price_volume_curve(self, orders):
        if not orders:
            return []
        orders = list(orders)
        curve = []
        cum_vol, cum_cost, order_idx = 0.0, 0.0, 0
        for target_vol in range(SPACING, 5000 + SPACING, SPACING):
            while cum_vol < target_vol and order_idx < len(orders):
                price, size = orders[order_idx]
                fill = min(size, target_vol - cum_vol)
                cum_cost += price * fill
                cum_vol += fill
                if size <= target_vol - (cum_vol - fill):
                    order_idx += 1
                else:
                    orders[order_idx] = (price, size - fill)
            if cum_vol < target_vol:
                break
            curve.append(cum_cost / cum_vol)
        return curve

    def calculate(self) -> Optional[BRTIResult]:
        books = self._filter_exchanges()
        if not books:
            return None
        exchange_mids = {k: v.mid_price for k, v in books.items()}
        all_bids, all_asks, cap = self._apply_cap_and_consolidate(books)
        if not all_bids or not all_asks:
            return None
        bid_curve = self._price_volume_curve(all_bids)
        ask_curve = self._price_volume_curve(all_asks)
        if not bid_curve or not ask_curve:
            return None
        max_depth = min(len(bid_curve), len(ask_curve))
        if max_depth == 0:
            return None
        bid_curve, ask_curve = bid_curve[:max_depth], ask_curve[:max_depth]
        mid_curve = [(b + a) / 2 for b, a in zip(bid_curve, ask_curve)]
        spread_curve = []
        for i in range(max_depth):
            spread_curve.append(
                (ask_curve[i] - mid_curve[i]) /
                mid_curve[i] if mid_curve[i] > 0 else float('inf')
            )
        utilized_depth = 0
        for i in range(max_depth):
            if spread_curve[i] <= MID_SPREAD_THRESHOLD:
                utilized_depth = i + 1
            else:
                break
        utilized_depth = max(utilized_depth, SPACING)

        weights = [LAMBDA * math.exp(-LAMBDA * (i + 1) * SPACING)
                   for i in range(utilized_depth)]
        wsum = sum(weights)
        if wsum > 0:
            weights = [w / wsum for w in weights]

        brti_value = sum(mid_curve[i] * weights[i]
                         for i in range(utilized_depth))
        spread_bps = ((ask_curve[0] - bid_curve[0]) /
                      mid_curve[0]) * 10000 if mid_curve[0] > 0 else 0

        result = BRTIResult(
            timestamp=time.time(), value=brti_value, utilized_depth=utilized_depth,
            num_exchanges=len(books), exchange_mids=exchange_mids,
            spread_bps=spread_bps, order_size_cap=cap,
        )
        self.history.append(result)
        if len(self.history) > 900:
            self.history = self.history[-900:]
        return result


# ═════════════════════════════════════════════════════════
# Exchange WebSocket Feeds (unchanged from original)
# ═════════════════════════════════════════════════════════

class ExchangeFeed:
    def __init__(self, engine: BRTIEngine, name: str):
        self.engine = engine
        self.name = name
        self.connected = False

    async def _reconnect_loop(self, connect_fn, delay=5):
        while True:
            try:
                await connect_fn()
            except Exception as e:
                log.error(f"{self.name} disconnected: {e}")
                self.connected = False
                await asyncio.sleep(delay)


class CoinbaseFeed(ExchangeFeed):
    def __init__(self, engine):
        super().__init__(engine, "Coinbase")

    async def connect(self):
        await self._reconnect_loop(self._run)

    async def _run(self):
        import websockets
        async with websockets.connect(
            "wss://advanced-trade-ws.coinbase.com",
            ping_interval=30, max_size=10*1024*1024, close_timeout=5,
        ) as ws:
            await ws.send(json.dumps({
                "type": "subscribe", "product_ids": ["BTC-USD"], "channel": "level2",
            }))
            self.connected = True
            log.info(f"{self.name} connected")
            bids, asks = {}, {}
            async for raw in ws:
                data = json.loads(raw)
                for event in data.get("events", []):
                    for u in event.get("updates", []):
                        price, size = float(u.get("price_level", 0)), float(
                            u.get("new_quantity", 0))
                        side, key = u.get("side", ""), u.get("price_level", "")
                        if side == "bid":
                            bids.pop(key, None) if size == 0 else bids.update(
                                {key: (price, size)})
                        elif side == "offer":
                            asks.pop(key, None) if size == 0 else asks.update(
                                {key: (price, size)})
                if bids or asks:
                    self.engine.update_order_book(
                        "coinbase",
                        sorted(bids.values(), key=lambda x: -x[0])[:500],
                        sorted(asks.values(), key=lambda x: x[0])[:500],
                    )


class KrakenFeed(ExchangeFeed):
    def __init__(self, engine):
        super().__init__(engine, "Kraken")

    async def connect(self):
        await self._reconnect_loop(self._run)

    async def _run(self):
        import websockets
        async with websockets.connect("wss://ws.kraken.com/v2", ping_interval=30, max_size=10*1024*1024) as ws:
            await ws.send(json.dumps({
                "method": "subscribe",
                "params": {"channel": "book", "symbol": ["BTC/USD"], "depth": 100},
            }))
            self.connected = True
            log.info(f"{self.name} connected")
            bids, asks = {}, {}
            async for raw in ws:
                data = json.loads(raw)
                if data.get("channel") != "book":
                    continue
                for entry in data.get("data", [{}]):
                    for b in entry.get("bids", []):
                        p, q = float(b["price"]), float(b["qty"])
                        bids.pop(p, None) if q == 0 else bids.update(
                            {p: (p, q)})
                    for a in entry.get("asks", []):
                        p, q = float(a["price"]), float(a["qty"])
                        asks.pop(p, None) if q == 0 else asks.update(
                            {p: (p, q)})
                if bids or asks:
                    self.engine.update_order_book(
                        "kraken", list(bids.values()), list(asks.values()))


class BitstampFeed(ExchangeFeed):
    def __init__(self, engine):
        super().__init__(engine, "Bitstamp")

    async def connect(self):
        await self._reconnect_loop(self._run)

    async def _run(self):
        import websockets
        async with websockets.connect("wss://ws.bitstamp.net", ping_interval=30, max_size=10*1024*1024) as ws:
            await ws.send(json.dumps({
                "event": "bts:subscribe", "data": {"channel": "order_book_btcusd"},
            }))
            self.connected = True
            log.info(f"{self.name} connected")
            async for raw in ws:
                data = json.loads(raw)
                if data.get("event") != "data":
                    continue
                book = data.get("data", {})
                bids = [(float(b[0]), float(b[1]))
                        for b in book.get("bids", [])]
                asks = [(float(a[0]), float(a[1]))
                        for a in book.get("asks", [])]
                if bids or asks:
                    self.engine.update_order_book("bitstamp", bids, asks)


class GeminiFeed(ExchangeFeed):
    def __init__(self, engine):
        super().__init__(engine, "Gemini")

    async def connect(self):
        await self._reconnect_loop(self._run)

    async def _run(self):
        import websockets
        async with websockets.connect("wss://api.gemini.com/v2/marketdata", ping_interval=30, max_size=10*1024*1024) as ws:
            await ws.send(json.dumps({
                "type": "subscribe", "subscriptions": [{"name": "l2", "symbols": ["BTCUSD"]}],
            }))
            self.connected = True
            log.info(f"{self.name} connected")
            bids, asks = {}, {}
            async for raw in ws:
                data = json.loads(raw)
                if data.get("type") != "l2_updates":
                    continue
                for change in data.get("changes", []):
                    if len(change) < 3:
                        continue
                    side, price, size = change[0], float(
                        change[1]), float(change[2])
                    if side == "buy":
                        bids.pop(price, None) if size == 0 else bids.update(
                            {price: (price, size)})
                    elif side == "sell":
                        asks.pop(price, None) if size == 0 else asks.update(
                            {price: (price, size)})
                if bids and asks:
                    self.engine.update_order_book(
                        "gemini", list(bids.values()), list(asks.values()))


# ═════════════════════════════════════════════════════════
# Kalshi API Poller (async version of your second script)
# ═════════════════════════════════════════════════════════

KALSHI_BASE_URL = 'https://api.elections.kalshi.com/trade-api/v2'
KALSHI_SERIES = 'KXBTC15M'


class KalshiPoller:
    """Async poller for Kalshi KXBTC15M market data."""

    def __init__(self, market_state: KalshiMarketState):
        self.state = market_state
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            )
        return self._session

    async def _fetch(self, url, params=None):
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    log.debug(f"Kalshi API {resp.status}: {url}")
                    return None
        except Exception as e:
            log.debug(f"Kalshi API error: {e}")
            return None

    async def poll_loop(self):
        """Poll Kalshi every 10 seconds for the active market."""
        log.info("Kalshi poller starting...")
        while True:
            try:
                data = await self._fetch(
                    f"{KALSHI_BASE_URL}/markets",
                    params={'series_ticker': KALSHI_SERIES,
                            'status': 'open', 'limit': 1}
                )

                if data and data.get('markets'):
                    market = data['markets'][0]
                    new_ticker = market['ticker']

                    # Detect market rollover
                    if self.state.ticker and new_ticker != self.state.ticker:
                        log.info(
                            f"Kalshi market rolled: {self.state.ticker} → {new_ticker}")

                    self.state.ticker = new_ticker
                    self.state.is_active = True
                    self.state.volume = market.get('volume', 0)

                    # Parse open/close times
                    open_str = market.get('open_time', '')
                    close_str = market.get('close_time', '') or market.get(
                        'expiration_time', '')
                    if open_str:
                        try:
                            self.state.open_time = datetime.fromisoformat(
                                open_str.replace('Z', '+00:00')).timestamp()
                        except Exception:
                            pass
                    if close_str:
                        try:
                            self.state.close_time = datetime.fromisoformat(
                                close_str.replace('Z', '+00:00')).timestamp()
                        except Exception:
                            pass

                    # Extract probability
                    yes_bid_raw = market.get('yes_bid_dollars')
                    yes_ask_raw = market.get('yes_ask_dollars')
                    if yes_bid_raw is not None and yes_ask_raw is not None:
                        yes_bid = float(yes_bid_raw)
                        yes_ask = float(yes_ask_raw)
                    else:
                        yes_bid = market.get('yes_bid', 0) / 100.0
                        yes_ask = market.get('yes_ask', 0) / 100.0

                    self.state.yes_bid = yes_bid
                    self.state.yes_ask = yes_ask
                    self.state.p_up = (yes_bid + yes_ask) / 2
                    self.state.p_down = 1.0 - self.state.p_up
                    self.state.last_poll = time.time()

                else:
                    self.state.is_active = False

            except Exception as e:
                log.debug(f"Kalshi poll error: {e}")

            await asyncio.sleep(10)

    async def check_resolution(self, ticker: str) -> Optional[str]:
        """Check if a market resolved. Returns 'UP', 'DOWN', or None."""
        data = await self._fetch(f"{KALSHI_BASE_URL}/markets/{ticker}")
        if data and data.get('market'):
            result = data['market'].get('result', '')
            if result == 'yes':
                return 'UP'
            elif result == 'no':
                return 'DOWN'
        return None


# ═════════════════════════════════════════════════════════
# Paper Trading Engine
# ═════════════════════════════════════════════════════════

class PaperTrader:
    """
    Simulated trading on Kalshi KXBTC15M binary markets.

    Strategy:
    - Uses BRTI to compute a fair P(UP) based on price movement from window start
    - Compares against Kalshi market P(UP)
    - Trades when |edge| > threshold, sized by fractional Kelly criterion
    - Positions resolve when the 15-minute market expires
    """

    def __init__(self, engine: BRTIEngine, kalshi_state: KalshiMarketState,
                 kalshi_poller: KalshiPoller, edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
                 csv_log: Optional[str] = None):
        self.engine = engine
        self.kalshi = kalshi_state
        self.poller = kalshi_poller
        self.edge_threshold = edge_threshold
        self.csv_log = csv_log

        self.bankroll = BANKROLL
        self.open_positions: list[PaperPosition] = []
        self.closed_positions: list[PaperPosition] = []
        self.next_position_id = 1
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0

        # Track per-market window start price
        self._window_start_prices: dict[str, float] = {}
        self._last_ticker = ""
        self._last_trade_time = 0.0
        self._cooldown_sec = 30  # Don't re-trade same market within 30s

        # Initialize CSV log
        if self.csv_log:
            if not os.path.isfile(self.csv_log):
                with open(self.csv_log, 'w', newline='') as f:
                    csv.writer(f).writerow([
                        'Timestamp', 'Action', 'Side', 'Size', 'Entry_Price',
                        'BRTI', 'Fair_P_UP', 'Kalshi_P_UP', 'Edge',
                        'Market_Ticker', 'PnL', 'Cumulative_PnL', 'Bankroll',
                        'Outcome', 'Window_Start', 'Exit_Reason'
                    ])

    def _log_trade(self, pos: PaperPosition, action: str):
        if not self.csv_log:
            return
        try:
            with open(self.csv_log, 'a', newline='') as f:
                csv.writer(f).writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    action, pos.side, pos.size,
                    f"{pos.entry_price:.4f}",
                    f"{pos.brti_at_entry:.2f}",
                    f"{pos.fair_p_at_entry:.4f}",
                    f"{pos.kalshi_p_at_entry:.4f}",
                    f"{pos.edge_at_entry:.4f}",
                    pos.market_ticker,
                    f"{pos.pnl:.2f}" if pos.pnl is not None else "",
                    f"{self.total_pnl:.2f}",
                    f"{self.bankroll:.2f}",
                    pos.outcome or "",
                    f"{pos.window_start_price:.2f}",
                    pos.exit_reason or "",
                ])
        except Exception as e:
            log.error(f"CSV log error: {e}")

    def compute_fair_p_up(self) -> Optional[dict]:
        """
        Derive fair P(UP) from BRTI price movement within the current window.

        Uses the same logic as KalshiSignalGenerator but returns structured data.
        """
        if not self.engine.history:
            return None

        current = self.engine.history[-1]
        ticker = self.kalshi.ticker
        if not ticker:
            return None

        # Get or set window start price for this market
        if ticker not in self._window_start_prices:
            self._window_start_prices[ticker] = current.value
            log.info(f"Window start for {ticker}: ${current.value:,.2f}")

        start_price = self._window_start_prices[ticker]
        pct_change = (current.value - start_price) / start_price

        # 30s momentum
        momentum = 0.0
        if len(self.engine.history) >= 30:
            momentum = (
                current.value - self.engine.history[-30].value) / self.engine.history[-30].value

        # 60s volatility
        vol = 0.0
        if len(self.engine.history) >= 60:
            prices = [h.value for h in self.engine.history[-60:]]
            returns = [(prices[i] - prices[i-1]) / prices[i-1]
                       for i in range(1, len(prices))]
            vol = float(np.std(returns)) if returns else 0.0

        # Sigmoid model: map price change to probability
        if abs(pct_change) < 0.00005:
            fair_p_up = 0.50
        else:
            z = pct_change / 0.001  # Normalize: 0.1% move ≈ 1 unit
            fair_p_up = 1.0 / (1.0 + math.exp(-0.8 * z))

        # Time decay: as window progresses, price matters more (sharper sigmoid)
        if self.kalshi.open_time and self.kalshi.close_time:
            elapsed_frac = (time.time() - self.kalshi.open_time) / (
                self.kalshi.close_time - self.kalshi.open_time)
            elapsed_frac = max(0, min(1, elapsed_frac))
            # Steeper sigmoid later in the window (price trend is more locked in)
            steepness = 0.8 + 1.2 * elapsed_frac  # 0.8 → 2.0
            if abs(pct_change) >= 0.00005:
                z = pct_change / 0.001
                fair_p_up = 1.0 / (1.0 + math.exp(-steepness * z))

        return {
            "fair_p_up": fair_p_up,
            "pct_change": pct_change,
            "momentum_30s": momentum,
            "vol_60s": vol,
            "brti_price": current.value,
            "start_price": start_price,
            "elapsed_frac": elapsed_frac if self.kalshi.open_time and self.kalshi.close_time else 0,
        }

    def _kelly_size(self, edge: float, entry_price: float) -> int:
        """
        Compute position size using fractional Kelly criterion.

        For a binary option at price p with fair probability q:
          Full Kelly = (q * (1-p) - (1-q) * p) / ((1-p) * p)
                     = (q/p - (1-q)/(1-p)) simplified
        We use quarter-Kelly for safety.
        """
        if entry_price <= 0 or entry_price >= 1:
            return 0

        # q = our fair probability, p = market price
        if edge > 0:
            q = entry_price + edge  # We think UP is underpriced
            p = entry_price
        else:
            q = (1 - entry_price) + abs(edge)  # We think DOWN is underpriced
            p = 1 - entry_price

        q = max(0.01, min(0.99, q))
        p = max(0.01, min(0.99, p))

        kelly = (q * (1 - p) - (1 - q) * p) / ((1 - p) * p)
        kelly = max(0, kelly)

        # Fractional Kelly
        bet_fraction = kelly * KELLY_FRACTION

        # Convert to number of $1 contracts
        size = int(self.bankroll * bet_fraction / entry_price)
        size = max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, size))

        # Don't bet more than 10% of bankroll
        max_risk = self.bankroll * 0.10
        if size * entry_price > max_risk:
            size = int(max_risk / entry_price)

        return max(0, size)

    def evaluate_and_trade(self):
        """Called every second. Check for edge and potentially open a position."""
        if not self.kalshi.is_active:
            return None

        # Don't trade if we already have max positions or recently traded
        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return None
        if time.time() - self._last_trade_time < self._cooldown_sec:
            return None

        # Already have a position in this market?
        for pos in self.open_positions:
            if pos.market_ticker == self.kalshi.ticker:
                return None

        # Compute fair value
        analysis = self.compute_fair_p_up()
        if analysis is None:
            return None

        fair_p_up = analysis["fair_p_up"]
        kalshi_p_up = self.kalshi.p_up
        edge = fair_p_up - kalshi_p_up

        # Check edge threshold
        if abs(edge) < self.edge_threshold:
            return None

        # High volatility = reduce confidence
        if analysis["vol_60s"] > 0.0005:
            effective_edge = edge * 0.6  # Discount edge in high-vol
            if abs(effective_edge) < self.edge_threshold:
                return None

        # Momentum confirmation bonus (don't require it, just note it)
        momentum_confirms = (
            (analysis["momentum_30s"] > 0 and fair_p_up > 0.5) or
            (analysis["momentum_30s"] < 0 and fair_p_up < 0.5)
        )

        # Determine trade side
        if edge > 0:
            side = "UP"
            entry_price = self.kalshi.yes_ask  # We buy YES at the ask
        else:
            side = "DOWN"
            # We buy NO at (1 - yes_bid)
            entry_price = 1.0 - self.kalshi.yes_bid

        if entry_price <= 0.01 or entry_price >= 0.99:
            return None  # Don't trade at extreme prices

        size = self._kelly_size(edge, entry_price)
        if size < MIN_POSITION_SIZE:
            return None

        # Open the position
        pos = PaperPosition(
            id=self.next_position_id,
            side=side,
            entry_price=entry_price,
            size=size,
            market_ticker=self.kalshi.ticker,
            brti_at_entry=analysis["brti_price"],
            window_start_price=analysis["start_price"],
            entry_time=time.time(),
            edge_at_entry=edge,
            fair_p_at_entry=fair_p_up,
            kalshi_p_at_entry=kalshi_p_up,
        )
        self.next_position_id += 1
        self.open_positions.append(pos)
        self._last_trade_time = time.time()

        self._log_trade(pos, "OPEN")
        return pos

    async def check_resolutions(self):
        """Check if any open positions' markets have resolved."""
        for pos in self.open_positions[:]:
            outcome = await self.poller.check_resolution(pos.market_ticker)
            if outcome is None:
                continue

            pos.outcome = outcome
            pos.exit_time = time.time()
            pos.exit_reason = "MARKET_RESOLVED"

            # P&L: Binary pays $1 if correct, $0 if not
            if pos.side == outcome:
                # Won: receive $1 per contract, paid entry_price per contract
                pos.pnl = pos.size * (1.0 - pos.entry_price)
                pos.exit_price = 1.0
                self.win_count += 1
            else:
                # Lost: receive $0, lose entry_price per contract
                pos.pnl = -pos.size * pos.entry_price
                pos.exit_price = 0.0
                self.loss_count += 1

            self.bankroll += pos.pnl
            self.total_pnl += pos.pnl
            self.open_positions.remove(pos)
            self.closed_positions.append(pos)
            self._log_trade(pos, "CLOSE")

            log.info(
                f"Position #{pos.id} resolved: {pos.side} → {outcome} | "
                f"PnL: ${pos.pnl:+.2f} | Cumulative: ${self.total_pnl:+.2f}"
            )


# ═════════════════════════════════════════════════════════
# Live Terminal Display (enhanced with trading info)
# ═════════════════════════════════════════════════════════

def render_frame(result: BRTIResult, engine: BRTIEngine,
                 kalshi: KalshiMarketState, trader: PaperTrader):
    now = datetime.now().strftime("%H:%M:%S")
    W = 68

    # 1-minute change
    if len(engine.history) >= 60:
        old = engine.history[-60].value
        delta = result.value - old
        pct = delta / old * 100
        change_str = (f"\033[92m▲ +${delta:,.2f} (+{pct:.3f}%)\033[0m" if delta >= 0
                      else f"\033[91m▼ −${abs(delta):,.2f} (−{abs(pct):.3f}%)\033[0m")
    else:
        change_str = f"\033[90mwait {60 - len(engine.history)}s...\033[0m"

    # Fair P(UP) from BRTI
    analysis = trader.compute_fair_p_up()
    fair_p_up = analysis["fair_p_up"] if analysis else 0.5
    pct_change = analysis["pct_change"] if analysis else 0.0
    edge = fair_p_up - kalshi.p_up if kalshi.is_active else 0.0

    # Exchange status
    ex_names = ["bitstamp", "coinbase", "gemini", "kraken"]
    ex_lines = []
    for name in ex_names:
        if name in result.exchange_mids:
            mid = result.exchange_mids[name]
            ex_lines.append(f"  \033[92m●\033[0m {name:<12} ${mid:>12,.2f}")
        else:
            ex_lines.append(f"  \033[91m●\033[0m {name:<12} {'offline':>13}")

    # Edge coloring
    if abs(edge) >= trader.edge_threshold:
        edge_color = "\033[92m" if edge > 0 else "\033[91m"
        edge_str = f"{edge_color}{edge:+.1%} ★ TRADEABLE\033[0m"
    elif abs(edge) >= 0.03:
        edge_str = f"\033[93m{edge:+.1%}\033[0m"
    else:
        edge_str = f"\033[90m{edge:+.1%}\033[0m"

    # Trading stats
    total_trades = trader.win_count + trader.loss_count
    win_rate = (trader.win_count / total_trades *
                100) if total_trades > 0 else 0
    pnl_color = "\033[92m" if trader.total_pnl >= 0 else "\033[91m"

    # Time remaining in window
    time_left = ""
    if kalshi.close_time:
        remaining = max(0, kalshi.close_time - time.time())
        mins, secs = divmod(int(remaining), 60)
        time_left = f"{mins}m{secs:02d}s"

    bar = "─" * W
    buf = ["\033[H\033[J"]
    buf.append(f"  ┌{bar}┐\n")
    buf.append(f"  │{'BRTI + KALSHI PAPER TRADER':^{W}}│\n")
    buf.append(f"  ├{bar}┤\n")

    # BRTI Price
    price_str = f"BRTI: ${result.value:,.2f}"
    buf.append(f"  │\033[1m{price_str:^{W}}\033[0m│\n")
    buf.append(f"  │{'   1m Δ: ' + change_str}\033[0m\033[{W + 3}G│\n")

    buf.append(f"  ├{bar}┤\n")

    # Kalshi market info
    if kalshi.is_active:
        kalshi_line1 = f"  Market: {kalshi.ticker}   Vol: {kalshi.volume}   TTL: {time_left}"
        kalshi_line2 = f"  Kalshi P(UP): {kalshi.p_up:.1%}  (bid {kalshi.yes_bid:.2f} / ask {kalshi.yes_ask:.2f})"
        kalshi_line3 = f"  BRTI  P(UP):  {fair_p_up:.1%}  (BTC {pct_change:+.4%} from open)"
        kalshi_line4 = f"  Edge:         {edge_str}"
    else:
        kalshi_line1 = "  Kalshi: No active market"
        kalshi_line2 = ""
        kalshi_line3 = ""
        kalshi_line4 = ""

    buf.append(f"  │{'  KALSHI SIGNAL':<{W}}│\n")
    buf.append(f"  │{kalshi_line1}\033[0m\033[{W + 3}G│\n")
    if kalshi_line2:
        buf.append(f"  │{kalshi_line2}\033[0m\033[{W + 3}G│\n")
        buf.append(f"  │{kalshi_line3}\033[0m\033[{W + 3}G│\n")
        buf.append(f"  │{kalshi_line4}\033[0m\033[{W + 3}G│\n")

    buf.append(f"  ├{bar}┤\n")

    # Trading stats
    buf.append(f"  │{'  PAPER TRADING':<{W}}│\n")
    bank_line = f"  Bankroll: ${trader.bankroll:,.2f}   PnL: {pnl_color}${trader.total_pnl:+,.2f}\033[0m"
    buf.append(f"  │{bank_line}\033[0m\033[{W + 3}G│\n")
    stats_line = f"  Trades: {total_trades}  W/L: {trader.win_count}/{trader.loss_count}  WR: {win_rate:.0f}%  Open: {len(trader.open_positions)}"
    buf.append(f"  │{stats_line}\033[0m\033[{W + 3}G│\n")

    # Show open positions
    for pos in trader.open_positions:
        pos_line = f"  ├ #{pos.id} {pos.side} x{pos.size} @{pos.entry_price:.2f} | {pos.market_ticker}"
        buf.append(f"  │{pos_line}\033[0m\033[{W + 3}G│\n")

    buf.append(f"  ├{bar}┤\n")

    # Exchange status
    stats = (f" Depth:{result.utilized_depth:>3} │ Sprd:{result.spread_bps:>5.1f}bp │"
             f" Cap:{result.order_size_cap:>6.1f} │ Src:{result.num_exchanges}/4 │ {now}")
    buf.append(f"  │{stats:<{W}}│\n")
    buf.append(f"  ├{bar}┤\n")
    for el in ex_lines:
        buf.append(f"  │{el}\033[0m\033[{W + 3}G│\n")

    buf.append(f"  └{bar}┘\n")
    buf.append(
        f"  \033[90mλ={LAMBDA:.3f}  Edge threshold: {trader.edge_threshold:.0%}  Ctrl+C to quit\033[0m\n")

    _sys.stdout.write("".join(buf))
    _sys.stdout.flush()


# ═════════════════════════════════════════════════════════
# Main Loop
# ═════════════════════════════════════════════════════════

async def main_loop(engine: BRTIEngine, kalshi_state: KalshiMarketState,
                    trader: PaperTrader):
    """Core loop: calculate BRTI, evaluate trades, check resolutions every second."""
    log.info("Waiting for exchange data...")
    await asyncio.sleep(5)
    logging.disable(logging.CRITICAL)
    _sys.stdout.write("\033[?25l\033[2J")
    _sys.stdout.flush()

    resolution_check_interval = 30  # Check resolutions every 30s
    last_resolution_check = 0

    try:
        while True:
            result = engine.calculate()
            if result:
                # Evaluate and potentially open a trade
                new_pos = trader.evaluate_and_trade()
                if new_pos:
                    # Re-enable logging briefly for trade notification
                    pass

                # Check resolutions periodically
                if time.time() - last_resolution_check > resolution_check_interval:
                    if trader.open_positions:
                        await trader.check_resolutions()
                    last_resolution_check = time.time()

                render_frame(result, engine, kalshi_state, trader)

            await asyncio.sleep(CALC_INTERVAL_SEC)
    finally:
        _sys.stdout.write("\033[?25h\n")
        _sys.stdout.flush()
        logging.disable(logging.NOTSET)


async def main():
    parser = argparse.ArgumentParser(description="BRTI + Kalshi Paper Trader")
    parser.add_argument("--aggressive", action="store_true",
                        help="Use lower edge threshold (3%% instead of 5%%)")
    parser.add_argument("--log", type=str, default="paper_trades.csv",
                        help="CSV file for trade log (default: paper_trades.csv)")
    parser.add_argument("--edge", type=float, default=None,
                        help="Custom edge threshold (e.g. 0.04 for 4%%)")
    args = parser.parse_args()

    edge_threshold = (
        args.edge if args.edge is not None
        else AGGRESSIVE_EDGE_THRESHOLD if args.aggressive
        else DEFAULT_EDGE_THRESHOLD
    )

    engine = BRTIEngine()
    kalshi_state = KalshiMarketState()
    kalshi_poller = KalshiPoller(kalshi_state)
    trader = PaperTrader(
        engine, kalshi_state, kalshi_poller,
        edge_threshold=edge_threshold,
        csv_log=args.log,
    )

    feeds = [CoinbaseFeed(engine), KrakenFeed(engine),
             BitstampFeed(engine), GeminiFeed(engine)]

    log.info(f"Starting BRTI + Kalshi Paper Trader")
    log.info(f"  Edge threshold: {edge_threshold:.0%}")
    log.info(f"  Bankroll: ${BANKROLL:,.0f}")
    log.info(f"  Kelly fraction: {KELLY_FRACTION}")
    log.info(f"  Trade log: {args.log}")
    log.info(f"  Exchanges: {[f.name for f in feeds]}")

    tasks = [asyncio.create_task(f.connect()) for f in feeds]
    tasks.append(asyncio.create_task(kalshi_poller.poll_loop()))
    tasks.append(asyncio.create_task(main_loop(engine, kalshi_state, trader)))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\033[?25h\n")
        print("Shutting down paper trader.")
