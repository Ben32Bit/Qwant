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

    # Return distribution shape
    skewness: Optional[float] = None
    excess_kurtosis: Optional[float] = None

    # Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
    # Probability in [0,1] that positive Sharpe is not due to overfitting/chance
    deflated_sharpe: Optional[float] = None

    # Benchmark-relative (optional — null if no benchmark)
    beta: Optional[float] = None
    alpha: Optional[float] = None
    r_squared: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None
    treynor: Optional[float] = None
    up_capture: Optional[float] = None
    down_capture: Optional[float] = None


class RollingMetrics(BaseModel):
    sharpe: Optional[list[TimeSeriesPoint]] = None       # 252-day rolling Sharpe
    volatility: Optional[list[TimeSeriesPoint]] = None   # 60-day rolling vol
    beta: Optional[list[TimeSeriesPoint]] = None         # 126-day rolling beta


class BacktestResult(BaseModel):
    equity_curve: list[TimeSeriesPoint]
    benchmark_curve: Optional[list[TimeSeriesPoint]] = None
    drawdown_series: list[DrawdownPoint]
    monthly_returns: MonthlyReturns
    metrics: PortfolioMetrics
    weight_history: Optional[list[dict]] = None       # [{date, ticker: weight, ...}]
    rebalance_dates: Optional[list[str]] = None       # dates when weights were reset
    rolling_metrics: Optional[RollingMetrics] = None
    correlation_matrix: Optional[dict] = None  # {ticker: {ticker: float}}
    fx_curves: Optional[dict[str, list[TimeSeriesPoint]]] = None  # {currency: [{date, value}]}
    # Populated by rotation backtests — shows which tickers were held each window
    holding_schedule: Optional[list[dict]] = None  # [{window_start, window_end, label, tickers: [str]}]
    # Fama-French Five-Factor decomposition (Fama & French, 2015)
    ff5_decomposition: Optional[dict] = None
