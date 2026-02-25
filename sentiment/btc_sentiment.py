"""
Bitcoin Sentiment Analysis & Trading Signal Generator
=====================================================

Scrapes Reddit every 15 minutes, runs FinBERT sentiment analysis, and
emits a LONG / SHORT / HOLD signal written to CSV.

Design decisions
----------------
* **FinBERT** (ProsusAI/finbert) is purpose-built for financial text and
  outperforms generic sentiment models on market-related language.
* **asyncio + aiohttp** for concurrent HTTP calls so the scrape phase
  finishes quickly.
* **Batched inference** on the GPU/CPU keeps transformer throughput high
  while bounding memory usage via a configurable batch size.
* Scores are persisted to a flat CSV so any downstream trading bot can
  simply tail / poll the file — no database dependency required.
* Reddit's public JSON endpoints are used — no API key required.

Usage
-----
    # 1. Install deps
    pip install -r requirements.txt

    # 2. Run (no API keys needed)
    python btc_sentiment.py
"""

from __future__ import annotations

import asyncio
import csv
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Scraping parameters
POSTS_PER_SOURCE: int = 100
REDDIT_SUBREDDITS: list[str] = ["Bitcoin", "CryptoCurrency", "CryptoMarkets", "BitcoinMarkets"]

# Sentiment thresholds — tune these based on back-testing
POSITIVE_THRESHOLD: float = 0.25
NEGATIVE_THRESHOLD: float = -0.25

# Loop interval (seconds)
CYCLE_INTERVAL: int = 15 * 60  # 15 minutes — no API quota to worry about

# Model config
MODEL_NAME: str = "ProsusAI/finbert"
BATCH_SIZE: int = 32

# Output
OUTPUT_DIR: Path = Path(__file__).resolve().parent
CSV_FILE: Path = OUTPUT_DIR / "btc_sentiment_history.csv"

# Reddit public JSON endpoints (no API key required)
REDDIT_USER_AGENT: str = "btc-sentiment-bot/1.0"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "btc_sentiment.log"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading (runs once at startup)
# ---------------------------------------------------------------------------

logger.info("Loading FinBERT model (%s) …", MODEL_NAME)
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
_model.eval()

# Move to GPU if available
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model.to(_device)
logger.info("Model loaded on %s", _device)

# FinBERT label mapping: index 0 = positive, 1 = negative, 2 = neutral
_LABEL_MAP = {0: 1.0, 1: -1.0, 2: 0.0}


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Strip URLs, mentions, hashtag symbols, and excess whitespace."""
    text = re.sub(r"http\S+|www\.\S+", "", text)       # URLs
    text = re.sub(r"@\w+", "", text)                    # @mentions
    text = re.sub(r"#", "", text)                       # hashtag symbol (keep word)
    text = re.sub(r"\n+", " ", text)                    # newlines
    text = re.sub(r"\s{2,}", " ", text)                 # multi-space
    return text.strip()


# ---------------------------------------------------------------------------
# Reddit fetcher (public JSON endpoints — no API key required)
# ---------------------------------------------------------------------------

async def fetch_reddit_posts(session: aiohttp.ClientSession) -> list[str]:
    """
    Fetch recent posts from Bitcoin-related subreddits using Reddit's
    public JSON endpoints (append .json to any subreddit URL).

    No API key needed.  Rate limit is ~10 req/min per IP which is fine
    since we only hit 4 subreddits once every 15 minutes.
    """
    headers = {"User-Agent": REDDIT_USER_AGENT}
    per_sub = max(1, POSTS_PER_SOURCE // len(REDDIT_SUBREDDITS))
    texts: list[str] = []

    for sub in REDDIT_SUBREDDITS:
        url = f"https://www.reddit.com/r/{sub}/new.json"
        params = {"limit": min(per_sub, 100)}

        try:
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", 60))
                    logger.warning("Reddit rate-limited on r/%s — sleeping %ds", sub, retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                if resp.status == 403:
                    logger.warning("Reddit returned 403 for r/%s — skipping", sub)
                    continue
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ClientError as exc:
            logger.error("Reddit request failed for r/%s: %s", sub, exc)
            continue

        for post in data.get("data", {}).get("children", []):
            title = post["data"].get("title", "")
            selftext = post["data"].get("selftext", "")
            combined = f"{title}. {selftext}" if selftext else title
            texts.append(clean_text(combined))

    logger.info("Fetched %d Reddit posts", len(texts))
    return texts[:POSTS_PER_SOURCE]


# ---------------------------------------------------------------------------
# Sentiment analysis (FinBERT batched inference)
# ---------------------------------------------------------------------------

def analyze_sentiment(texts: list[str]) -> float:
    """
    Run FinBERT on a list of texts and return a mean sentiment score
    in [-1, 1].  Positive = bullish, negative = bearish.

    Uses batched inference to keep throughput high without blowing memory.
    Returns 0.0 (neutral) when the input list is empty.
    """
    if not texts:
        return 0.0

    scores: list[float] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        encodings = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(_device)

        with torch.no_grad():
            logits = _model(**encodings).logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()

        scores.extend(_LABEL_MAP[p] for p in preds)

    mean_score = sum(scores) / len(scores)
    return round(mean_score, 4)


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signal(score: float) -> str:
    """Map the sentiment score to a trading signal."""
    if score > POSITIVE_THRESHOLD:
        return "LONG"
    if score < NEGATIVE_THRESHOLD:
        return "SHORT"
    return "HOLD"


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

def save_results(
    timestamp: str,
    reddit_score: float,
    signal: str,
    reddit_count: int,
) -> None:
    """Append one row to the sentiment history CSV."""
    file_exists = CSV_FILE.exists()

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "reddit_score",
                "signal",
                "reddit_posts",
            ])
        writer.writerow([
            timestamp,
            reddit_score,
            signal,
            reddit_count,
        ])

    logger.info("Results saved to %s", CSV_FILE)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_cycle() -> dict:
    """
    Execute a single scrape → analyse → signal cycle.

    Returns a dict with the score and signal so callers (or a future
    trading bot integration) can consume the result programmatically.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=== Cycle start: %s UTC ===", timestamp)

    async with aiohttp.ClientSession() as session:
        reddit_texts = await fetch_reddit_posts(session)

    reddit_score = analyze_sentiment(reddit_texts)
    signal = generate_signal(reddit_score)

    save_results(
        timestamp=timestamp,
        reddit_score=reddit_score,
        signal=signal,
        reddit_count=len(reddit_texts),
    )

    logger.info(
        "Reddit=%.4f (%d posts) | Signal=%s",
        reddit_score, len(reddit_texts), signal,
    )

    return {
        "timestamp": timestamp,
        "reddit_score": reddit_score,
        "signal": signal,
    }


def _seconds_until_next_quarter() -> float:
    """Return seconds until the next :00 / :15 / :30 / :45 boundary."""
    now = datetime.now(timezone.utc)
    current_minute = now.minute
    # Next quarter-hour boundary
    next_quarter = (current_minute // 15 + 1) * 15
    if next_quarter == 60:
        # Roll over to the top of the next hour
        target = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        target = now.replace(minute=next_quarter, second=0, microsecond=0)
    return (target - now).total_seconds()


async def main() -> None:
    """
    Run the sentiment loop aligned to clock quarter-hours (:00, :15, :30, :45)
    so results line up with Kalshi 15-minute Bitcoin markets.
    """
    logger.info(
        "Starting BTC sentiment loop (aligned to :00/:15/:30/:45, thresholds=+%.2f/-%.2f)",
        POSITIVE_THRESHOLD, abs(NEGATIVE_THRESHOLD),
    )

    while True:
        # Wait until the next quarter-hour boundary
        wait = _seconds_until_next_quarter()
        logger.info("Waiting %.0f seconds until next quarter-hour …", wait)
        await asyncio.sleep(wait)

        try:
            await run_cycle()
        except Exception:
            logger.exception("Unhandled error in cycle — will retry next interval")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested — exiting cleanly")
