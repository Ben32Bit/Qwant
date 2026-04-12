from pydantic import BaseModel
from typing import Optional


class TickerWindowResult(BaseModel):
    ticker: str
    metric_value: float
    rank: int


class ScreenerWindow(BaseModel):
    window_start: str
    window_end: str
    label: str          # human-readable, e.g. "2023 Q1" or "Jan 2024"
    rankings: list[TickerWindowResult]
    winner: Optional[str] = None


class ScreenerResult(BaseModel):
    windows: list[ScreenerWindow]
    metric: str
    metric_label: str
    window_freq: str
    tickers: list[str]
    top_n: int
    screen_description: Optional[str] = None


class ScreenRequest(BaseModel):
    tickers: list[str]
    start_date: str
    end_date: str
    window_freq: str    # monthly | quarterly | annually | weekly
    metric: str         # return | sharpe | volatility | max_drawdown | momentum_3m
    top_n: int = 3
    benchmark: str = "SPY"
    screen_description: Optional[str] = None


class RotationBacktestRequest(BaseModel):
    screen_result: ScreenerResult
    initial_capital: float = 10_000
    benchmark: str = "SPY"
    top_n_held: int = 1   # how many top-ranked assets to hold each period (equal weight)
