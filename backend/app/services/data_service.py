import yfinance as yf
import pandas as pd
from threading import Lock
from app.utils.cache import get_price_cache, price_cache_key

_lock = Lock()


def fetch_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.
    Returns a DataFrame with dates as index and tickers as columns.
    Caches results for 24 hours.
    """
    cache = get_price_cache()
    key = price_cache_key(tickers, start_date, end_date)

    with _lock:
        if key in cache:
            return cache[key].copy()

    # Download from yfinance
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError(f"No price data returned for tickers: {tickers}")

    # Extract 'Close' prices (auto_adjust=True gives adjusted close as 'Close')
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Keep only requested tickers (in case yfinance drops some)
    available = [t for t in tickers if t in prices.columns]
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"No data available for tickers: {missing}")

    prices = prices[available].dropna(how="all")

    with _lock:
        cache[key] = prices.copy()

    return prices


def search_tickers(query: str) -> list[dict]:
    """
    Simple ticker search using yfinance. Returns list of {ticker, name, exchange}.
    """
    try:
        ticker = yf.Ticker(query.upper())
        info = ticker.fast_info
        return [{"ticker": query.upper(), "name": getattr(info, "longName", query.upper()), "exchange": ""}]
    except Exception:
        return []
