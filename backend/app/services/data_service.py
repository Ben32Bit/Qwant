import yfinance as yf
import pandas as pd
import numpy as np
from threading import Lock
from datetime import date
import logging
import time

from app.utils.cache import get_price_cache, price_cache_key
from app.services.price_store import (
    load_prices, store_prices, tickers_needing_fetch,
)

logger = logging.getLogger(__name__)

_lock = Lock()
TRADING_DAYS = 252
_YF_MAX_RETRIES = 3
_YF_RETRY_DELAY = 2  # seconds


# ── yfinance download with retry ──────────────────────────────────────────────

def _yf_download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted closes from yfinance with retry on failure."""
    for attempt in range(_YF_MAX_RETRIES):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                if attempt < _YF_MAX_RETRIES - 1:
                    logger.warning("yfinance returned empty data, retrying (%d/%d)…", attempt + 1, _YF_MAX_RETRIES)
                    time.sleep(_YF_RETRY_DELAY)
                    continue
            return raw
        except Exception as e:
            if attempt < _YF_MAX_RETRIES - 1:
                logger.warning("yfinance error: %s — retrying (%d/%d)…", e, attempt + 1, _YF_MAX_RETRIES)
                time.sleep(_YF_RETRY_DELAY)
            else:
                raise
    return pd.DataFrame()


def _extract_closes(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract the Close column from a raw yfinance DataFrame (handles MultiIndex)."""
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker — flat DataFrame
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return prices.dropna(how="all")


# ── Main fetch (L1: TTLCache → L2: SQLite → L3: yfinance) ────────────────────

def fetch_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers.

    Cache hierarchy:
      L1 — in-memory TTLCache (24h, per-process)
      L2 — SQLite price store (persistent across restarts)
      L3 — yfinance (only fetches what's missing from L2)

    Raises ValueError if no data can be found for any ticker.
    Logs a warning and skips tickers that are unavailable.
    """
    cache = get_price_cache()
    key = price_cache_key(tickers, start_date, end_date)

    # L1: in-memory hit
    with _lock:
        if key in cache:
            return cache[key].copy()

    today = date.today().isoformat()
    effective_end = min(end_date, today)

    # L2: SQLite — determine what's already stored
    to_fetch, fetch_start, fetch_end = tickers_needing_fetch(tickers, start_date, effective_end)

    # L3: yfinance — fetch only the missing gaps
    if to_fetch:
        logger.info("Fetching %s from yfinance (%s → %s)", to_fetch, fetch_start, fetch_end)
        raw = _yf_download(to_fetch, fetch_start, fetch_end)
        if not raw.empty:
            new_prices = _extract_closes(raw, to_fetch)
            if not new_prices.empty:
                store_prices(new_prices)  # persist to SQLite
        elif to_fetch:
            logger.warning("yfinance returned no data for %s", to_fetch)

    # Load full requested range from SQLite (now includes newly fetched data)
    prices = load_prices(tickers, start_date, effective_end)

    if prices.empty:
        raise ValueError(
            f"No price data available for tickers: {tickers}. "
            "Check that all tickers are valid US-listed symbols."
        )

    # Log any tickers that came back empty
    available = [t for t in tickers if t in prices.columns and prices[t].notna().any()]
    missing = [t for t in tickers if t not in available]
    if missing:
        logger.warning("No data found for tickers (skipping): %s", missing)
    if not available:
        raise ValueError(f"No price data available for any of: {tickers}")

    prices = prices[available].dropna(how="all")

    # L1: populate in-memory cache
    with _lock:
        cache[key] = prices.copy()

    return prices


def fetch_prices_partial(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Like fetch_prices but never raises — silently drops tickers with no data.
    Used by the AI research tool where candidate tickers may be invalid.
    """
    try:
        return fetch_prices(tickers, start_date, end_date)
    except ValueError:
        # Try one by one to salvage whatever we can
        frames = []
        for t in tickers:
            try:
                df = fetch_prices([t], start_date, end_date)
                frames.append(df)
            except Exception:
                pass
        if not frames:
            raise ValueError(f"No price data returned for any of: {tickers}")
        result = pd.concat(frames, axis=1)
        result = result.loc[:, ~result.columns.duplicated()]
        return result.dropna(how="all")


# ── AI research tool ──────────────────────────────────────────────────────────

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


# ── Ticker search ─────────────────────────────────────────────────────────────

def search_tickers(query: str) -> list[dict]:
    try:
        ticker = yf.Ticker(query.upper())
        info = ticker.fast_info
        return [{"ticker": query.upper(), "name": getattr(info, "longName", query.upper()), "exchange": ""}]
    except Exception:
        return []
