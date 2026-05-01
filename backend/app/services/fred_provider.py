"""
FREDProvider — Macro data from FRED.

Fetch strategy (in priority order):
  1. fredapi.Fred(api_key=FRED_API_KEY) — authenticated, higher rate limits,
     returns data up to yesterday. Requires FRED_API_KEY in environment.
  2. pandas_datareader (source="fred") — anonymous public API, no key required,
     occasionally rate-limited or 1-2 days stale.

Series fetched (all daily, forward-filled over weekends/holidays):
  T10Y2Y  : 10Y-2Y Treasury yield curve spread (recession/expansion signal)
  BAA10Y  : BAA corporate bond spread over 10Y Treasury (credit risk premium)
  DFII10  : 10Y TIPS real yield (real interest rate signal)
  DFF     : Federal Funds Rate (overnight lending rate)

Add to .env:
  FRED_API_KEY=your_key_here    # free at https://fred.stlouisfed.org/docs/api/api_key.html

Returns scalar floats for the most recent observation plus rolling percentile
ranks vs. the prior 1,260 trading days (~5 years).

Caching
-------
24-hour in-process cache keyed by calendar date. Railway restarts daily.

Fallback
--------
If both fetch methods fail, _NEUTRAL long-run averages are returned so
inference always produces a result (model predictions revert to unconditional mean).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── FRED series IDs ───────────────────────────────────────────────────────────

FRED_SERIES = {
    "yield_curve_10y2y": "T10Y2Y",    # percent; daily; can be negative
    "credit_spread_baa": "BAA10Y",    # percent; daily; always positive
    "real_yield_10y":    "DFII10",    # percent; daily; can be negative
    "fed_funds_rate":    "DFF",       # percent; daily
}

# Long-run historical averages used when FRED is unreachable.
_NEUTRAL: dict[str, float] = {
    "yield_curve_10y2y":    0.60,   # avg 10Y-2Y since 1990 (~60 bps)
    "credit_spread_baa":    2.20,   # avg BAA-10Y since 1990
    "real_yield_10y":       0.50,   # avg TIPS yield 2003-2024
    "fed_funds_rate":       2.50,   # avg DFF since 1990
    "credit_pct_rank":      0.50,
    "yield_curve_pct_rank": 0.50,
    "real_yield_pct_rank":  0.50,
    "available":            False,
}

CACHE_TTL_SECONDS = 86_400   # 24 hours
HISTORY_YEARS     = 7        # calendar years to download for percentile ranks
LOOKBACK_DAYS     = 1_260    # ~5Y trading days for rank window

_cache: dict[str, tuple[float, dict]] = {}   # {date_str: (timestamp, result)}


# ── Public interface ──────────────────────────────────────────────────────────

def get_macro_features(as_of_date: Optional[str] = None) -> dict:
    """
    Return the latest macro feature dict, fetching from FRED if the cache is stale.

    Fetch chain: fredapi (authenticated) → pandas_datareader (anonymous) → neutral fallback.

    Returns
    -------
    dict with keys:
        yield_curve_10y2y, credit_spread_baa, real_yield_10y, fed_funds_rate
        credit_pct_rank, yield_curve_pct_rank, real_yield_pct_rank
        available : bool  (False if neutral fallback was used)
        source    : str   ("fredapi" | "pandas_datareader" | "neutral_fallback")
    """
    key = as_of_date or date.today().isoformat()

    if key in _cache:
        ts, result = _cache[key]
        if time.time() - ts < CACHE_TTL_SECONDS:
            return result

    api_key = os.environ.get("FRED_API_KEY", "").strip()
    result  = None

    # 1. Direct FRED REST API — no extra packages, just requests/urllib (preferred)
    if api_key:
        try:
            result = _fetch_via_requests(key, api_key)
        except Exception as exc:
            logger.warning("FREDProvider: REST API failed (%s) — trying fredapi", exc)

    # 2. fredapi library (if installed)
    if result is None and api_key:
        try:
            result = _fetch_via_fredapi(key, api_key)
        except Exception as exc:
            logger.warning("FREDProvider: fredapi failed (%s) — trying pandas_datareader", exc)

    # 3. pandas_datareader (anonymous, no key needed)
    if result is None:
        try:
            result = _fetch_via_pandas_datareader(key)
        except Exception as exc:
            logger.warning("FREDProvider: pandas_datareader failed (%s) — neutral fallback", exc)

    # 4. Final fallback: neutral long-run averages
    if result is None:
        result = {**_NEUTRAL, "source": "neutral_fallback"}

    _cache[key] = (time.time(), result)
    return result


def download_macro_history(start: str, end: str) -> pd.DataFrame:
    """
    Download FRED macro series for the training period.
    Returns DataFrame with columns = feature names, business-day index.

    Fetch chain: fredapi → pandas_datareader → neutral constants.
    """
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    df      = None

    # 1. Direct FRED REST API — no extra packages required (uses requests or urllib)
    if api_key:
        try:
            df = _history_via_requests(start, end, api_key)
            logger.info("FREDProvider: history fetched via FRED REST API")
        except Exception as exc:
            logger.warning("FREDProvider: REST API history failed (%s) — trying fredapi", exc)

    # 2. fredapi library (if installed)
    if df is None and api_key:
        try:
            df = _history_via_fredapi(start, end, api_key)
            logger.info("FREDProvider: history fetched via fredapi")
        except Exception as exc:
            logger.warning("FREDProvider: fredapi history failed (%s) — trying pandas_datareader", exc)

    # 3. pandas_datareader (anonymous)
    if df is None:
        try:
            df = _history_via_pandas_datareader(start, end)
            logger.info("FREDProvider: history fetched via pandas_datareader")
        except Exception as exc:
            logger.warning("FREDProvider: pandas_datareader history failed (%s) — neutral constants", exc)

    # 4. Final fallback: neutral constants
    if df is None:
        idx = pd.bdate_range(start, end)
        df  = pd.DataFrame(
            {k: v for k, v in _NEUTRAL.items() if k not in ("available", "source")},
            index=idx,
        )
        logger.warning("FREDProvider: using neutral macro constants for training")

    return df


# ── Direct FRED REST API (no external packages — uses requests or urllib) ─────
#
# FRED JSON API endpoint:
#   GET https://api.stlouisfed.org/fred/series/observations
#       ?series_id=T10Y2Y&api_key=KEY&observation_start=...&observation_end=...
#       &file_type=json
#
# This is the primary fetch path when FRED_API_KEY is set.
# No fredapi or pandas_datareader needed — requests is guaranteed via yfinance.

FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _fred_series_via_requests(series_id: str, start: str, end: str, api_key: str) -> pd.Series:
    """Fetch one FRED series as a daily pd.Series using requests or urllib."""
    url = (
        f"{FRED_API_BASE}?series_id={series_id}"
        f"&observation_start={start}&observation_end={end}"
        f"&api_key={api_key}&file_type=json"
    )
    try:
        import requests  # noqa: PLC0415
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except ImportError:
        import urllib.request, json as _json  # noqa: PLC0415
        with urllib.request.urlopen(url, timeout=30) as r:
            data = _json.loads(r.read())

    observations = data.get("observations", [])
    if not observations:
        raise ValueError(f"FRED API returned no observations for {series_id}")

    rows = {
        obs["date"]: float(obs["value"])
        for obs in observations
        if obs.get("value", ".") != "."
    }
    s = pd.Series(rows)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _fetch_via_requests(as_of_date: str, api_key: str) -> dict:
    end   = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
    start = (pd.Timestamp(as_of_date) - pd.DateOffset(years=HISTORY_YEARS)).strftime("%Y-%m-%d")

    series_map: dict[str, pd.Series] = {}
    for feat_name, series_id in FRED_SERIES.items():
        s = _fred_series_via_requests(series_id, start, end, api_key)
        series_map[feat_name] = s.ffill().dropna()

    return _build_result(series_map, source="fred_rest_api")


def _history_via_requests(start: str, end: str, api_key: str) -> pd.DataFrame:
    dl_start = (pd.Timestamp(start) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    raw: dict[str, pd.Series] = {}
    for feat_name, series_id in FRED_SERIES.items():
        raw[feat_name] = _fred_series_via_requests(series_id, dl_start, end, api_key)

    df = pd.DataFrame(raw).ffill()
    return _align_history(df, start, end)


# ── fredapi fetch (optional alternative) ──────────────────────────────────────

def _fetch_via_fredapi(as_of_date: str, api_key: str) -> dict:
    from fredapi import Fred  # noqa: PLC0415

    fred  = Fred(api_key=api_key)
    end   = pd.Timestamp(as_of_date)
    start = end - pd.DateOffset(years=HISTORY_YEARS)

    series_map: dict[str, pd.Series] = {}
    for feat_name, series_id in FRED_SERIES.items():
        s = fred.get_series(series_id,
                            observation_start=start.strftime("%Y-%m-%d"),
                            observation_end=end.strftime("%Y-%m-%d"))
        series_map[feat_name] = s.ffill().dropna()

    return _build_result(series_map, source="fredapi")


def _history_via_fredapi(start: str, end: str, api_key: str) -> pd.DataFrame:
    from fredapi import Fred  # noqa: PLC0415

    fred     = Fred(api_key=api_key)
    dl_start = (pd.Timestamp(start) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    raw: dict[str, pd.Series] = {}
    for feat_name, series_id in FRED_SERIES.items():
        raw[feat_name] = fred.get_series(series_id, observation_start=dl_start, observation_end=end)

    df = pd.DataFrame(raw).ffill()
    return _align_history(df, start, end)


# ── pandas_datareader fetch (anonymous, no key needed) ────────────────────────

def _fetch_via_pandas_datareader(as_of_date: str) -> dict:
    import pandas_datareader.data as web  # noqa: PLC0415

    end   = pd.Timestamp(as_of_date)
    start = end - pd.DateOffset(years=HISTORY_YEARS)

    series_ids = list(FRED_SERIES.values())
    df = web.DataReader(series_ids, "fred", start, end)
    df = df.ffill().dropna(how="all")

    if df.empty:
        raise ValueError("FRED returned empty DataFrame")

    series_map = {
        feat: df[sid].dropna()
        for feat, sid in FRED_SERIES.items()
        if sid in df.columns
    }
    return _build_result(series_map, source="pandas_datareader")


def _history_via_pandas_datareader(start: str, end: str) -> pd.DataFrame:
    import pandas_datareader.data as web  # noqa: PLC0415

    dl_start = (pd.Timestamp(start) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    series_ids = list(FRED_SERIES.values())
    df = web.DataReader(series_ids, "fred", dl_start, end)
    df = df.ffill()
    df = df.rename(columns={v: k for k, v in FRED_SERIES.items()})
    return _align_history(df, start, end)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _pct_rank(series: pd.Series, window: int = LOOKBACK_DAYS) -> float:
    tail = series.dropna().iloc[-min(window, len(series)):]
    if len(tail) < 2:
        return 0.5
    v = float(tail.iloc[-1])
    return float((tail < v).sum() / len(tail))


def _build_result(series_map: dict[str, pd.Series], source: str) -> dict:
    """Build the result dict from a map of {feature_name: pd.Series}."""
    def latest(name: str) -> float:
        s = series_map.get(name, pd.Series(dtype=float))
        if s.empty:
            return float(_NEUTRAL[name])
        v = float(s.iloc[-1])
        return v if not np.isnan(v) else float(_NEUTRAL[name])

    yield_curve   = latest("yield_curve_10y2y")
    credit_spread = latest("credit_spread_baa")
    real_yield    = latest("real_yield_10y")
    fed_funds     = latest("fed_funds_rate")

    return {
        "yield_curve_10y2y":    round(yield_curve,   3),
        "credit_spread_baa":    round(credit_spread, 3),
        "real_yield_10y":       round(real_yield,    3),
        "fed_funds_rate":       round(fed_funds,     3),
        "credit_pct_rank":      round(_pct_rank(series_map.get("credit_spread_baa", pd.Series())), 3),
        "yield_curve_pct_rank": round(_pct_rank(series_map.get("yield_curve_10y2y", pd.Series())), 3),
        "real_yield_pct_rank":  round(_pct_rank(series_map.get("real_yield_10y",    pd.Series())), 3),
        "available":            True,
        "source":               source,
    }


def _align_history(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Reindex to business days, add rolling percentile ranks."""
    df = df.reindex(pd.bdate_range(start, end), method="ffill")
    df = df.loc[start:end]

    for col in ["credit_spread_baa", "yield_curve_10y2y"]:
        if col in df.columns:
            rank_col = col.replace("credit_spread_baa", "credit_pct_rank") \
                          .replace("yield_curve_10y2y", "yield_curve_pct_rank")
            df[rank_col] = df[col].expanding().apply(
                lambda s: float((s < s.iloc[-1]).sum() / len(s)), raw=False
            )

    return df.fillna(0.0)
