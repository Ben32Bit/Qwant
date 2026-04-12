from pydantic import BaseModel
from typing import Optional


class TimeSeriesPoint(BaseModel):
    date: str
    value: float


class DrawdownPoint(BaseModel):
    date: str
    drawdown: float


class MonthlyReturns(BaseModel):
    # {year: {month_abbr: return}}
    data: dict[str, dict[str, Optional[float]]]


class PortfolioMetrics(BaseModel):
    # Return metrics
    total_return: float
    cagr: float
    best_year: Optional[dict] = None
    worst_year: Optional[dict] = None
    best_month: Optional[dict] = None
    worst_month: Optional[dict] = None

    # Risk metrics
    volatility: float
    max_drawdown: float
    max_drawdown_duration_days: int
    downside_deviation: float
    var_95: float
    cvar_95: float

    # Risk-adjusted
    sharpe: float
    sortino: float
    calmar: float

    # Benchmark-relative (optional — null if no benchmark)
    beta: Optional[float] = None
    alpha: Optional[float] = None
    r_squared: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None
    treynor: Optional[float] = None
    up_capture: Optional[float] = None
    down_capture: Optional[float] = None


class BacktestResult(BaseModel):
    equity_curve: list[TimeSeriesPoint]
    benchmark_curve: Optional[list[TimeSeriesPoint]] = None
    drawdown_series: list[DrawdownPoint]
    monthly_returns: MonthlyReturns
    metrics: PortfolioMetrics
    weight_history: Optional[list[dict]] = None
