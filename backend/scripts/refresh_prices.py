"""
Nightly price refresh script.

Downloads today's adjusted closes for popular tickers so they're
pre-warmed in SQLite before the first user request of the day.

Run after market close (e.g. 4:30 PM ET via Railway cron):
  python -m scripts.refresh_prices

Railway cron syntax (in railway.toml):
  [cron]
  schedule = "30 21 * * 1-5"   # 21:30 UTC = 4:30 PM ET on weekdays
  command = "python -m scripts.refresh_prices"
"""

import sys
import os
import logging
from datetime import date, timedelta
from pathlib import Path

# Add backend root to path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.services.price_store import init_db, store_prices, get_coverage
from app.services.data_service import _yf_download, _extract_closes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger("refresh_prices")

# Tickers to pre-warm — top ETFs and their common combinations
POPULAR_TICKERS = [
    # Core benchmarks
    "SPY", "QQQ", "IWM", "VTI", "VOO",
    # Bonds
    "BND", "AGG", "TLT", "IEF", "SHY", "LQD", "HYG",
    # International
    "EFA", "EEM", "VEA", "VWO", "IEFA",
    # Alternatives
    "GLD", "IAU", "SLV", "VNQ", "PDBC",
    # Factors
    "MTUM", "QUAL", "VLUE", "USMV", "SIZE",
    # Sectors
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLU", "XLY", "XLP",
    # Leveraged (popular in backtesting)
    "UPRO", "SSO", "TMF", "TQQQ",
    # Popular individual stocks
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "JPM",
]

LOOKBACK_YEARS = 15  # store up to 15 years of history


def main():
    init_db()
    logger.info("Refreshing prices for %d tickers…", len(POPULAR_TICKERS))

    today = date.today().isoformat()
    start = (date.today().replace(year=date.today().year - LOOKBACK_YEARS)).isoformat()

    # Check coverage — only fetch what's missing or stale
    coverage = get_coverage(POPULAR_TICKERS)
    stale = []
    missing = []

    for ticker in POPULAR_TICKERS:
        if ticker not in coverage:
            missing.append(ticker)
        else:
            _, max_dt = coverage[ticker]
            yesterday = (date.today() - timedelta(days=1)).isoformat()
            if max_dt < yesterday:
                stale.append(ticker)

    logger.info("Missing: %d, Stale: %d, Up-to-date: %d",
                len(missing), len(stale), len(POPULAR_TICKERS) - len(missing) - len(stale))

    # Fetch missing tickers (full history)
    if missing:
        logger.info("Fetching full history for %d new tickers…", len(missing))
        raw = _yf_download(missing, start, today)
        if not raw.empty:
            prices = _extract_closes(raw, missing)
            n = store_prices(prices)
            logger.info("Stored %d rows for new tickers", n)

    # Update stale tickers (recent data only)
    if stale:
        fetch_from = (date.today() - timedelta(days=7)).isoformat()
        logger.info("Updating %d stale tickers from %s…", len(stale), fetch_from)
        raw = _yf_download(stale, fetch_from, today)
        if not raw.empty:
            prices = _extract_closes(raw, stale)
            n = store_prices(prices)
            logger.info("Updated %d rows for stale tickers", n)

    logger.info("Refresh complete.")


if __name__ == "__main__":
    main()
