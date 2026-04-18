"""
Pydantic models for the /api/forecast endpoint.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class ForecastAsset(BaseModel):
    ticker: str
    weight: float


class ForecastRequest(BaseModel):
    equity_curve: list[dict]          # [{date: str, value: float}, ...]
    assets: list[ForecastAsset]
    start_date: str                   # YYYY-MM-DD
    end_date: str                     # YYYY-MM-DD (last historical date)
    horizon_days: int = 252
    n_paths: int = 1000
    methods: list[str] = ["monte_carlo", "garch", "factor", "hmm", "var", "lstm"]
    ff5_decomposition: Optional[dict] = None


class ForecastBand(BaseModel):
    dates: list[str]
    p5: list[float]
    p25: list[float]
    p50: list[float]
    p75: list[float]
    p95: list[float]


class MethodResult(BaseModel):
    method: str
    label: str
    color: str
    forecast: Optional[ForecastBand] = None
    metadata: dict = {}
    error: Optional[str] = None
    compute_ms: int = 0


class ForecastResponse(BaseModel):
    forecast_start: str
    forecast_end: str
    historical_end_value: float
    results: list[MethodResult]
    news_context: Optional[dict] = None    # per-ticker GDELT headlines for browser FinBERT
    edgar_context: Optional[dict] = None   # per-ticker 10-K/10-Q excerpts for browser FinBERT
    tier2_context: Optional[dict] = None   # reddit + trends aggregate for method metadata
    regime_probs: Optional[dict] = None    # 4-state regime probabilities for meta-ensemble
    ensemble_weights: Optional[dict] = None  # regime-conditional method weights
