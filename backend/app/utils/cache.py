import time
from cachetools import TTLCache
from threading import Lock

# Price data cache: key = (ticker, start, end), TTL = 24 hours
_price_cache: TTLCache = TTLCache(maxsize=500, ttl=86400)
_price_lock = Lock()

# AI response cache: key = normalized prompt hash, TTL = 1 hour
_ai_cache: TTLCache = TTLCache(maxsize=200, ttl=3600)
_ai_lock = Lock()


def get_price_cache() -> TTLCache:
    return _price_cache


def get_ai_cache() -> TTLCache:
    return _ai_cache


def price_cache_key(tickers: list[str], start_date: str, end_date: str) -> str:
    return f"{','.join(sorted(tickers))}|{start_date}|{end_date}"
