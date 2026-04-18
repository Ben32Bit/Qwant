"""
GoogleTrendsProvider — search interest z-score for portfolio tickers.

Uses pytrends (unofficial Google Trends scraper). Rate-limited by Google;
requests are cached aggressively (24h). Falls back gracefully when pytrends
is not installed or when Google rate-limits.

Feature returned
----------------
interest_7d      : float [0, 100] — mean relative search interest in last 7 days
interest_90d_avg : float          — 90-day baseline mean
zscore_7d        : float          — (interest_7d − interest_90d_avg) / std_90d
                                   Positive = above-average public attention
available        : bool

References
----------
Preis, T., Moat, H.S. & Stanley, H.E. (2013). Quantifying Trading Behavior in
  Financial Markets Using Google Trends. Scientific Reports, 3, 1684.
  https://doi.org/10.1038/srep01684

Da, Z., Engelberg, J. & Gao, P. (2011). In Search of Attention. Journal of
  Finance, 66(5), 1461–1499. https://doi.org/10.1111/j.1540-6261.2011.01679.x
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 86_400   # 24 hours — aggressive; Google rate-limits hard

_cache: dict[str, tuple[float, dict]] = {}

_SKIP_TICKERS = frozenset({
    "SPY", "QQQ", "GLD", "TLT", "AGG", "IWM", "VTI", "BND",
    "XLK", "XLF", "XLE", "EFA", "EEM",
})


def get_trends_context(tickers: list[str], as_of_date: str) -> dict:
    """
    Return Google Trends z-scores for each ticker.

    Returns
    -------
    dict:
      {ticker: {"interest_7d": float, "zscore_7d": float, "available": bool}, ...}
      "portfolio_summary": {"available": bool}
    """
    investable = [t for t in tickers if t not in _SKIP_TICKERS]
    if not investable:
        return {"portfolio_summary": {"available": False}}

    cache_key = f"{','.join(sorted(investable))}:{as_of_date}"
    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if time.time() - ts < CACHE_TTL_SECONDS:
            return cached

    result: dict = {}
    any_ok = False

    # pytrends is optional; skip gracefully if not installed
    try:
        from pytrends.request import TrendReq  # noqa: PLC0415
    except ImportError:
        logger.debug("TrendsProvider: pytrends not installed — skipping")
        result["portfolio_summary"] = {"available": False}
        return result

    try:
        # Process in batches of 5 (pytrends limit per request)
        for i in range(0, len(investable), 5):
            batch = investable[i:i + 5]
            batch_result = _fetch_batch(TrendReq, batch, as_of_date)
            result.update(batch_result)
            if any(v.get("available") for v in batch_result.values()):
                any_ok = True
    except Exception as exc:
        logger.warning("TrendsProvider: batch fetch failed: %s", exc)

    result["portfolio_summary"] = {"available": any_ok}
    _cache[cache_key] = (time.time(), result)
    return result


def _fetch_batch(TrendReq, tickers: list[str], as_of_date: str) -> dict:
    try:
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        pytrends.build_payload(tickers, timeframe="today 3-m", geo="US")
        df = pytrends.interest_over_time()
    except Exception as exc:
        logger.debug("TrendsProvider pytrends request failed: %s", exc)
        return {t: {"interest_7d": 0.0, "zscore_7d": 0.0, "available": False} for t in tickers}

    if df.empty:
        return {t: {"interest_7d": 0.0, "zscore_7d": 0.0, "available": False} for t in tickers}

    result = {}
    for ticker in tickers:
        if ticker not in df.columns:
            result[ticker] = {"interest_7d": 0.0, "zscore_7d": 0.0, "available": False}
            continue
        series = df[ticker].dropna().astype(float)
        if len(series) < 14:
            result[ticker] = {"interest_7d": 0.0, "zscore_7d": 0.0, "available": False}
            continue
        recent_7  = float(series.iloc[-5:].mean())   # last ~5 weekly points ≈ 7d
        baseline  = series.iloc[:-5]
        std_b     = float(baseline.std()) if len(baseline) > 1 else 1.0
        mean_b    = float(baseline.mean())
        zscore    = (recent_7 - mean_b) / (std_b + 1e-9)
        result[ticker] = {
            "interest_7d":      round(recent_7, 1),
            "interest_90d_avg": round(mean_b, 1),
            "zscore_7d":        round(zscore, 2),
            "available":        True,
        }
    return result
