import yfinance as yf
import pandas as pd
import numpy as np
from threading import Lock
from app.utils.cache import get_price_cache, price_cache_key

_lock = Lock()
TRADING_DAYS = 252


def fetch_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.
    Returns a DataFrame with dates as index and tickers as columns.
    Caches results for 24 hours. Silently drops unavailable tickers.
    """
    cache = get_price_cache()
    key = price_cache_key(tickers, start_date, end_date)

    with _lock:
        if key in cache:
            return cache[key].copy()

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

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    available = [t for t in tickers if t in prices.columns]
    missing = [t for t in tickers if t not in prices.columns]
    if not available:
        raise ValueError(f"No price data available for any of the requested tickers: {tickers}")
    if missing:
        import logging
        logging.getLogger(__name__).warning(f"Tickers not found in Yahoo Finance response, skipping: {missing}")

    prices = prices[available].dropna(how="all")

    with _lock:
        cache[key] = prices.copy()

    return prices


def fetch_prices_partial(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Like fetch_prices but silently drops tickers with no data instead of raising.
    Used by the AI research tool where some candidate tickers may be invalid.
    """
    cache = get_price_cache()
    key = price_cache_key(tickers, start_date, end_date)

    with _lock:
        if key in cache:
            return cache[key].copy()

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError(f"No price data returned for any of: {tickers}")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")

    with _lock:
        cache[key] = prices.copy()

    return prices


def get_asset_statistics_for_ai(
    tickers: list[str],
    start_date: str,
    end_date: str,
    benchmark: str = "SPY",
) -> dict:
    """
    Compute per-asset statistics for the AI research tool.
    Returns annualised return, volatility, Sharpe, max drawdown,
    correlation to benchmark, and the full correlation matrix.
    """
    all_tickers = list(set(tickers + [benchmark]))

    try:
        prices = fetch_prices_partial(all_tickers, start_date, end_date)
    except ValueError as e:
        return {"error": str(e)}

    returns = prices.pct_change().dropna()
    bm_ret = returns[benchmark] if benchmark in returns.columns else None

    stats: dict = {}
    for ticker in tickers:
        if ticker not in returns.columns:
            stats[ticker] = {"available": False, "reason": "no data from Yahoo Finance"}
            continue

        r = returns[ticker].dropna()
        if r.empty:
            stats[ticker] = {"available": False, "reason": "no overlapping data in date range"}
            continue
        cum = (1 + r).cumprod()
        total_ret = float(cum.iloc[-1] - 1)
        ann_ret = float((1 + total_ret) ** (TRADING_DAYS / max(len(r), 1)) - 1)
        ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))
        sharpe = round(ann_ret / ann_vol, 3) if ann_vol > 0 else 0.0
        max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())

        entry: dict = {
            "available": True,
            "annual_return_pct": round(ann_ret * 100, 2),
            "annual_volatility_pct": round(ann_vol * 100, 2),
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "total_return_pct": round(total_ret * 100, 2),
        }

        if bm_ret is not None:
            entry[f"correlation_to_{benchmark}"] = round(float(r.corr(bm_ret)), 3)

        stats[ticker] = entry

    # Full pairwise correlation matrix
    available = [t for t in tickers if t in returns.columns]
    if len(available) > 1:
        corr = returns[available].corr().round(3)
        stats["_correlation_matrix"] = corr.to_dict()

    stats["_meta"] = {
        "start": start_date,
        "end": end_date,
        "trading_days": len(returns),
        "benchmark": benchmark,
    }

    return stats


def search_tickers(query: str) -> list[dict]:
    try:
        ticker = yf.Ticker(query.upper())
        info = ticker.fast_info
        return [{"ticker": query.upper(), "name": getattr(info, "longName", query.upper()), "exchange": ""}]
    except Exception:
        return []
