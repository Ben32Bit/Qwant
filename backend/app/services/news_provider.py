"""
NewsProvider — GDELT 2.0 DOC API headlines for portfolio tickers.

Uses direct HTTP requests to GDELT's free public API (no authentication needed,
no packages required). Returns article titles for client-side FinBERT scoring.

Caching
-------
12-hour in-process cache keyed by ticker list + date. GDELT data updates hourly,
but 12h is sufficient for daily portfolio-level sentiment signals.

Fallback
--------
Any network or parsing failure returns {"headlines": [], "available": False} for
that ticker — inference continues without sentiment.

References
----------
Leetaru, K. & Schrodt, P.A. (2013). GDELT: Global Data on Events, Language,
  and Tone, 1979-2012. International Studies Association Annual Conference.
  https://data.gdeltproject.org/documentation/ISA.2013.GDELT.pdf

Preis, T., Moat, H.S. & Stanley, H.E. (2013). Quantifying Trading Behavior in
  Financial Markets Using Google Trends. Scientific Reports, 3, 1684.
  https://doi.org/10.1038/srep01684
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta

logger = logging.getLogger(__name__)

GDELT_DOC_API     = "https://api.gdeltproject.org/api/v2/doc/doc"
CACHE_TTL_SECONDS = 43_200   # 12 hours
MAX_HEADLINES     = 15       # per ticker per fetch

_cache: dict[str, tuple[float, dict]] = {}

# ETF/macro tickers — no meaningful company headlines
_SKIP_TICKERS = frozenset({
    "SPY", "QQQ", "GLD", "TLT", "AGG", "IWM", "VTI", "BND",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE",
    "EFA", "EEM", "IAU", "SHY", "IEF", "DIA", "MDY", "VNQ",
})

# GDELT indexes article text, not ticker symbols.
# Map tickers to more natural search terms for better recall.
_QUERY_MAP: dict[str, str] = {
    "AAPL": "Apple earnings stock",
    "MSFT": "Microsoft earnings stock",
    "GOOGL": "Alphabet Google earnings stock",
    "GOOG":  "Alphabet Google earnings stock",
    "AMZN": "Amazon earnings stock",
    "NVDA": "Nvidia earnings stock",
    "META": "Meta Facebook earnings stock",
    "TSLA": "Tesla earnings stock",
    "BRK.B": "Berkshire Hathaway stock",
    "JPM":  "JPMorgan Chase earnings stock",
    "BAC":  "Bank of America earnings stock",
    "WMT":  "Walmart earnings stock",
    "XOM":  "ExxonMobil earnings stock",
    "CVX":  "Chevron earnings stock",
}


def get_news_context(tickers: list[str], weights: dict[str, float], as_of_date: str) -> dict:
    """
    Fetch recent headlines from GDELT for each non-ETF ticker.

    Parameters
    ----------
    tickers   : list of ticker symbols
    weights   : weight dict (not used for fetching, kept for future relevance weighting)
    as_of_date: YYYY-MM-DD — fetch articles from [as_of_date-7d, as_of_date]

    Returns
    -------
    dict:
      {ticker: {"headlines": [str, ...], "article_count": int, "available": bool}, ...}
      "portfolio_summary": {"total_articles": int, "available": bool}
    """
    investable = [t for t in tickers if t not in _SKIP_TICKERS]
    if not investable:
        return {"portfolio_summary": {"total_articles": 0, "available": False}}

    cache_key = f"{','.join(sorted(investable))}:{as_of_date}"
    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if time.time() - ts < CACHE_TTL_SECONDS:
            return cached

    end_dt   = date.fromisoformat(as_of_date)
    start_dt = end_dt - timedelta(days=7)
    start_s  = start_dt.strftime("%Y%m%d") + "000000"
    end_s    = end_dt.strftime("%Y%m%d")   + "235959"

    result: dict = {}
    total         = 0

    for ticker in investable:
        query = _QUERY_MAP.get(ticker, ticker)
        try:
            headlines = _fetch_headlines(query, start_s, end_s)
            result[ticker] = {"headlines": headlines, "article_count": len(headlines), "available": True}
            total += len(headlines)
        except Exception as exc:
            logger.debug("NewsProvider GDELT failed for %s: %s", ticker, exc)
            result[ticker] = {"headlines": [], "article_count": 0, "available": False}

    result["portfolio_summary"] = {
        "total_articles": total,
        "available":      any(v.get("available") for k, v in result.items() if k != "portfolio_summary"),
    }

    _cache[cache_key] = (time.time(), result)
    return result


def _fetch_headlines(query: str, start_s: str, end_s: str) -> list[str]:
    q_enc = query.replace(" ", "%20")
    url = (
        f"{GDELT_DOC_API}?query={q_enc}"
        f"&mode=artlist&maxrecords={MAX_HEADLINES}&format=json"
        f"&startdatetime={start_s}&enddatetime={end_s}&sourcelang=english"
    )
    headers = {"User-Agent": "PortfolioBacktester/1.0 (research; non-commercial)"}

    try:
        import requests  # noqa: PLC0415
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except ImportError:
        import urllib.request as _urllib  # noqa: PLC0415
        req = _urllib.Request(url, headers=headers)
        with _urllib.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())

    articles = data.get("articles") or []
    return [a["title"] for a in articles if a.get("title")]
