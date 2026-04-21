"""
Ticker feed — data source for the marquee at the top of the app.

Mixes three live sources into a single flat list the frontend just iterates:
  1. Latest prices + daily % change for a curated ticker list
  2. Broad-market news headlines from GDELT 2.0 DOC API
  3. Trending post titles from r/wallstreetbets / r/investing / r/stocks

All three are cached in-process (5 minutes) so repeated requests from the
scrolling banner don't hammer upstream APIs.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

from fastapi import APIRouter

from app.services.data_service import fetch_prices
from app.services.news_provider import _fetch_headlines
from app.services.reddit_provider import get_trending_posts

logger = logging.getLogger(__name__)

router = APIRouter()

# Curated tickers — popular + broad market coverage.
_TICKER_LIST = [
    "SPY", "QQQ", "DIA", "IWM", "VTI",          # Broad equity ETFs
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",    # Mega caps
    "AMZN", "TSLA", "AMD", "NFLX", "AVGO",      # More names
    "GLD", "TLT", "USO", "VIXY",                # Commodities / rates / vol
    "BTC-USD", "ETH-USD",                       # Crypto
]

_NEWS_QUERY = "stock market Wall Street Nasdaq Dow Jones earnings"

_CACHE_TTL_S = 300   # 5 minutes
_cache: dict[str, tuple[float, dict]] = {}


@router.get("/ticker/feed")
async def ticker_feed() -> dict:
    """
    Return the mixed scroll-ticker feed.

    Shape:
      {
        "items": [
          {"kind": "price",  "ticker": "AAPL", "price": 189.42, "change_pct": 1.2},
          {"kind": "news",   "headline": "…"},
          {"kind": "reddit", "title": "…", "subreddit": "wallstreetbets", "score": 342},
          ...
        ],
        "updated_at": "2026-04-21T14:32:15Z",
        "sources":    {"prices": int, "news": int, "reddit": int},
      }

    If any upstream source fails, the other two still populate — we never
    return an error; the UI just shows less variety.
    """
    now = time.time()
    cached = _cache.get("feed")
    if cached and now - cached[0] < _CACHE_TTL_S:
        return cached[1]

    prices = _build_price_items()
    news   = _build_news_items()
    reddit = _build_reddit_items()

    # Interleave so the marquee doesn't clump one source together.
    items: list[dict] = []
    idx = 0
    while idx < max(len(prices), len(news), len(reddit)):
        if idx < len(prices): items.append(prices[idx])
        if idx < len(news):   items.append(news[idx])
        if idx < len(reddit): items.append(reddit[idx])
        idx += 1

    payload = {
        "items": items,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "sources": {"prices": len(prices), "news": len(news), "reddit": len(reddit)},
    }
    _cache["feed"] = (now, payload)
    return payload


# ── Price items ───────────────────────────────────────────────────────────────

def _build_price_items() -> list[dict]:
    """
    Fetch last ~7 trading days for the curated list, compute last-vs-prev %
    change. We pull a week because a single 2-day window misses weekends /
    holidays. fetch_prices already uses the SQLite L2 cache so this is cheap.
    """
    try:
        end_dt   = date.today()
        start_dt = end_dt - timedelta(days=10)
        prices = fetch_prices(_TICKER_LIST, start_dt.isoformat(), end_dt.isoformat())
    except Exception as exc:
        logger.warning("Ticker feed: price fetch failed: %s", exc)
        return []

    out: list[dict] = []
    for ticker in _TICKER_LIST:
        if ticker not in prices.columns:
            continue
        series = prices[ticker].dropna()
        if len(series) < 2:
            continue
        last = float(series.iloc[-1])
        prev = float(series.iloc[-2])
        if prev <= 0:
            continue
        change_pct = (last / prev - 1.0) * 100.0
        out.append({
            "kind":       "price",
            "ticker":     ticker,
            "price":      round(last, 2),
            "change_pct": round(change_pct, 2),
        })
    return out


# ── News items ────────────────────────────────────────────────────────────────

def _build_news_items() -> list[dict]:
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=2)
    start_s  = start_dt.strftime("%Y%m%d") + "000000"
    end_s    = end_dt.strftime("%Y%m%d")   + "235959"
    try:
        headlines = _fetch_headlines(_NEWS_QUERY, start_s, end_s)
    except Exception as exc:
        logger.warning("Ticker feed: GDELT fetch failed: %s", exc)
        return []

    out: list[dict] = []
    seen: set[str] = set()
    for h in headlines:
        clean = (h or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append({"kind": "news", "headline": clean[:200]})
        if len(out) >= 12:
            break
    return out


# ── Reddit items ──────────────────────────────────────────────────────────────

def _build_reddit_items() -> list[dict]:
    try:
        posts = get_trending_posts(
            subreddits=["wallstreetbets", "investing", "stocks"],
            limit_per_sub=4,
        )
    except Exception as exc:
        logger.warning("Ticker feed: reddit trending failed: %s", exc)
        return []

    out: list[dict] = []
    for p in posts[:10]:
        out.append({
            "kind":      "reddit",
            "title":     p["title"],
            "subreddit": p["subreddit"],
            "score":     p["score"],
            "url":       p.get("url", ""),
        })
    return out
