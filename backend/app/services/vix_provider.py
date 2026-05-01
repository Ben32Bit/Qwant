"""
VIXProvider — VIX term structure from yfinance.

Symbols downloaded:
  ^VIX   : CBOE Volatility Index (30-day implied vol on SPX options)
  ^VIX3M : CBOE 3-Month Volatility Index (93-day implied vol)

Derived features
----------------
  vix_spot       : latest VIX closing level
  vix_pct_rank   : VIX percentile rank vs trailing 504 days (~2Y) — 0 = calm, 1 = panic
  vix_term_slope : VIX3M / VIX − 1  (positive = contango; negative = backwardation)
  vix_contango   : 1.0 if VIX3M > VIX else 0.0

Interpretation
--------------
  Contango (vix_term_slope > 0): market expects higher vol later than now.
    Common in calm regimes. Historically bullish for equities (Ang et al. 2006).
  Backwardation (vix_term_slope < 0): near-term fear exceeds long-term concern.
    Often coincides with corrections / high-stress periods.

Caching
-------
Same 24-hour in-process cache as fred_provider.py. Keyed by calendar date.

Fallback
--------
If yfinance returns no data (API changes, network issues), neutral values
are returned: vix_spot=20, pct_rank=0.50, slope=0.05 (mild contango).

References
----------
Ang, A., Hodrick, R.J., Xing, Y., & Zhang, X. (2006). The cross-section of
  volatility and expected returns. Journal of Finance, 61(1), 259–299.
  https://doi.org/10.1111/j.1540-6261.2006.00836.x
CBOE VIX White Paper: https://www.cboe.com/tradable_products/vix/vix_white_paper.pdf
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NEUTRAL: dict[str, float] = {
    "vix_spot":       20.0,   # long-run VIX mean ~20
    "vix_pct_rank":    0.50,
    "vix_term_slope":  0.05,  # mild contango
    "vix_contango":    1.0,
    "available":       False,
}

CACHE_TTL_SECONDS = 86_400   # 24 hours
LOOKBACK_TRADING  = 504      # ~2 years for percentile rank

_cache: dict[str, tuple[float, dict]] = {}


def get_vix_features(as_of_date: Optional[str] = None) -> dict:
    """
    Return the latest VIX term-structure features.

    Parameters
    ----------
    as_of_date : str, optional
        ISO date (YYYY-MM-DD). Used as cache key. Defaults to today.

    Returns
    -------
    dict with keys: vix_spot, vix_pct_rank, vix_term_slope, vix_contango, available
    """
    key = as_of_date or date.today().isoformat()

    if key in _cache:
        ts, result = _cache[key]
        if time.time() - ts < CACHE_TTL_SECONDS:
            return result

    try:
        result = _fetch_vix(key)
    except Exception as exc:
        logger.warning("VIXProvider: fetch failed (%s) — returning neutral values", exc)
        result = dict(_NEUTRAL)

    _cache[key] = (time.time(), result)
    return result


def _fetch_vix(as_of_date: str) -> dict:
    import yfinance as yf  # noqa: PLC0415

    end = pd.Timestamp(as_of_date) + pd.Timedelta(days=1)
    start = pd.Timestamp(as_of_date) - pd.DateOffset(years=3)

    data = yf.download(["^VIX", "^VIX3M"], start=start, end=end,
                       auto_adjust=True, progress=False)["Close"]

    if data.empty or "^VIX" not in data.columns:
        raise ValueError("yfinance returned no VIX data")

    data = data.ffill().dropna(how="all")

    vix     = data["^VIX"].dropna()
    vix3m   = data["^VIX3M"].dropna() if "^VIX3M" in data.columns else pd.Series(dtype=float)

    if vix.empty:
        raise ValueError("VIX series is empty")

    vix_spot = float(vix.iloc[-1])

    # Percentile rank vs trailing 2Y
    tail = vix.iloc[-LOOKBACK_TRADING:]
    vix_pct_rank = float((tail < vix_spot).sum() / len(tail))

    # Term slope (VIX3M / VIX − 1)
    if not vix3m.empty and len(vix3m) > 0:
        vix3m_spot = float(vix3m.iloc[-1])
        vix_term_slope = round(vix3m_spot / (vix_spot + 1e-9) - 1.0, 4)
        vix_contango   = 1.0 if vix3m_spot > vix_spot else 0.0
    else:
        vix_term_slope = _NEUTRAL["vix_term_slope"]
        vix_contango   = _NEUTRAL["vix_contango"]

    return {
        "vix_spot":       round(vix_spot,       2),
        "vix_pct_rank":   round(vix_pct_rank,   3),
        "vix_term_slope": round(vix_term_slope, 4),
        "vix_contango":   vix_contango,
        "available":      True,
    }


def download_vix_history(start: str, end: str) -> pd.DataFrame:
    """
    Download VIX history for training.
    Returns a DataFrame with columns [vix_spot, vix_pct_rank, vix_term_slope, vix_contango],
    business-day index, aligned to [start, end].

    """
    import yfinance as yf  # noqa: PLC0415

    data = yf.download(
        ["^VIX", "^VIX3M"],
        start=(pd.Timestamp(start) - pd.DateOffset(years=1)).strftime("%Y-%m-%d"),
        end=end,
        auto_adjust=True, progress=False,
    )["Close"].ffill()

    vix   = data.get("^VIX",   pd.Series(dtype=float))
    vix3m = data.get("^VIX3M", pd.Series(dtype=float))

    if vix.empty:
        # Construct neutral DataFrame over the date range
        idx = pd.bdate_range(start, end)
        return pd.DataFrame({
            "vix_spot":       _NEUTRAL["vix_spot"],
            "vix_pct_rank":   _NEUTRAL["vix_pct_rank"],
            "vix_term_slope": _NEUTRAL["vix_term_slope"],
            "vix_contango":   _NEUTRAL["vix_contango"],
        }, index=idx)

    # Rolling percentile rank (expanding window)
    vix_pct_rank = vix.expanding().apply(
        lambda s: float((s < s.iloc[-1]).sum() / len(s)), raw=False
    )

    # Term slope
    if not vix3m.empty:
        vix_term_slope = vix3m / (vix + 1e-9) - 1.0
        vix_contango   = (vix3m > vix).astype(float)
    else:
        vix_term_slope = pd.Series(_NEUTRAL["vix_term_slope"], index=vix.index)
        vix_contango   = pd.Series(_NEUTRAL["vix_contango"],   index=vix.index)

    df = pd.DataFrame({
        "vix_spot":       vix,
        "vix_pct_rank":   vix_pct_rank,
        "vix_term_slope": vix_term_slope,
        "vix_contango":   vix_contango,
    })

    df = df.reindex(pd.bdate_range(start, end), method="ffill")
    df = df.loc[start:end]
    return df.fillna(0.0)
