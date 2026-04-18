"""
RedditProvider — ticker mention velocity from r/wallstreetbets, r/investing, r/stocks.

Uses Reddit's public read-only JSON API (no authentication required).
No packages needed — plain urllib/requests.

Feature returned
----------------
mention_count_7d : int   — posts mentioning ticker in last 7 days across subreddits
avg_score_7d     : float — mean post score (upvotes − downvotes); proxy for engagement
available        : bool

Noise note (displayed in UI)
----------------------------
Reddit WSB sentiment reflects retail flow with high short-squeeze / meme-stock
manipulation risk. Treat as a contrarian or momentum signal, not fundamental.

References
----------
Bollen, J., Mao, H. & Zeng, X. (2011). Twitter mood predicts the stock market.
  Journal of Computational Science, 2(1), 1–8.
  https://doi.org/10.1016/j.jocs.2010.12.007

Kearney, C. & Liu, S. (2014). Textual sentiment in finance: A survey of methods
  and models. International Review of Financial Analysis, 33, 171–185.
  https://doi.org/10.1016/j.irfa.2014.02.006

Umar, Z. et al. (2021). The impact of COVID-19-related media coverage on the
  return and volatility connectedness of cryptocurrencies and fiat currencies.
  Technological Forecasting & Social Change, 172, 121025.
"""

from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 10_800   # 3 hours
SUBREDDITS        = ["wallstreetbets", "investing", "stocks"]

_cache: dict[str, tuple[float, dict]] = {}

_SKIP_TICKERS = frozenset({
    "SPY", "QQQ", "GLD", "TLT", "AGG", "IWM", "VTI", "BND",
    "XLK", "XLF", "XLE", "EFA", "EEM",
})


def get_reddit_context(tickers: list[str], as_of_date: str) -> dict:
    """
    Return 7-day mention counts and mean post scores per ticker.

    Returns
    -------
    dict:
      {ticker: {"mention_count_7d": int, "avg_score_7d": float, "available": bool}, ...}
      "portfolio_summary": {"total_mentions": int, "available": bool}
    """
    investable = [t for t in tickers if t not in _SKIP_TICKERS]
    if not investable:
        return {"portfolio_summary": {"total_mentions": 0, "available": False}}

    cache_key = f"{','.join(sorted(investable))}:{as_of_date}"
    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if time.time() - ts < CACHE_TTL_SECONDS:
            return cached

    result: dict = {}
    total = 0

    for ticker in investable:
        counts, scores = [], []
        for sub in SUBREDDITS:
            try:
                posts = _fetch_posts(sub, ticker)
                counts.append(len(posts))
                if posts:
                    scores.extend(p.get("score", 0) for p in posts)
            except Exception as exc:
                logger.debug("Reddit %s/%s failed: %s", sub, ticker, exc)

        mention_count = sum(counts)
        avg_score     = float(sum(scores) / len(scores)) if scores else 0.0
        result[ticker] = {
            "mention_count_7d": mention_count,
            "avg_score_7d":     round(avg_score, 1),
            "available":        bool(counts),
        }
        total += mention_count

    result["portfolio_summary"] = {
        "total_mentions": total,
        "available":      any(v.get("available") for k, v in result.items()
                              if k != "portfolio_summary"),
    }

    _cache[cache_key] = (time.time(), result)
    return result


def _fetch_posts(subreddit: str, query: str) -> list[dict]:
    url = (
        f"https://www.reddit.com/r/{subreddit}/search.json"
        f"?q={query}&restrict_sr=1&sort=new&t=week&limit=25"
    )
    headers = {
        "User-Agent": "PortfolioBacktester/1.0 (research; non-commercial)",
        "Accept": "application/json",
    }
    try:
        import requests  # noqa: PLC0415
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except ImportError:
        import urllib.request as _urllib  # noqa: PLC0415
        req = _urllib.Request(url, headers=headers)
        with _urllib.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())

    children = data.get("data", {}).get("children", [])
    return [c.get("data", {}) for c in children]
