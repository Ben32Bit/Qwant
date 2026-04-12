import pandas as pd
import numpy as np
from typing import Optional
from app.services.data_service import fetch_prices
from app.services.metrics import calculate_metrics, build_monthly_returns_grid
from app.models.portfolio import PortfolioInput
from app.models.backtest_result import BacktestResult, TimeSeriesPoint, DrawdownPoint, MonthlyReturns, PortfolioMetrics
from app.utils.constants import TRADING_DAYS_PER_YEAR
import os


RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))


def _get_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set:
    if freq == "none":
        return {index[0]}
    if freq == "daily":
        return set(index)

    freq_map = {
        "weekly": "W",
        "monthly": "ME",
        "quarterly": "QE",
        "annually": "YE",
    }
    period_freq = freq_map.get(freq, "QE")
    # Generate rebalance dates aligned to business calendar
    rebalance = pd.date_range(start=index[0], end=index[-1], freq=period_freq)
    # Map to actual trading days (find nearest trading day for each rebalance date)
    result = set()
    for rd in rebalance:
        # Find the first trading day >= rd
        candidates = index[index >= rd]
        if len(candidates) > 0:
            result.add(candidates[0])
    result.add(index[0])  # always rebalance on first day
    return result


def run_backtest(
    prices_df: pd.DataFrame,
    weights: dict[str, float],
    rebalance_freq: str,
    initial_capital: float = 10_000.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Core backtest loop.

    Returns:
        portfolio_values: pd.Series indexed by date
        weight_history: pd.DataFrame of weights over time
    """
    # Align to tickers that exist in prices
    tickers = [t for t in weights if t in prices_df.columns]
    prices = prices_df[tickers].copy()
    daily_returns = prices.pct_change().dropna()

    target_weights = pd.Series({t: weights[t] for t in tickers})
    rebalance_dates = _get_rebalance_dates(daily_returns.index, rebalance_freq)

    current_weights = target_weights.copy()
    portfolio_values = [initial_capital]
    weight_records = []

    for date in daily_returns.index:
        if date in rebalance_dates:
            current_weights = target_weights.copy()

        day_rets = daily_returns.loc[date]
        daily_ret = float((current_weights * day_rets).sum())
        new_value = portfolio_values[-1] * (1 + daily_ret)
        portfolio_values.append(new_value)

        # Drift weights
        new_weights = current_weights * (1 + day_rets)
        weight_sum = new_weights.sum()
        if weight_sum != 0:
            current_weights = new_weights / weight_sum
        else:
            current_weights = target_weights.copy()

        weight_records.append({"date": date.strftime("%Y-%m-%d"), **current_weights.to_dict()})

    portfolio_series = pd.Series(portfolio_values[1:], index=daily_returns.index)
    weight_history = pd.DataFrame(weight_records)

    return portfolio_series, weight_history


def run_full_backtest(portfolio_input: PortfolioInput) -> BacktestResult:
    """
    End-to-end backtest: fetch data → run backtest → compute metrics → build result.
    """
    tickers = [a.ticker for a in portfolio_input.assets]
    weights = {a.ticker: a.weight for a in portfolio_input.assets}
    benchmark = portfolio_input.benchmark

    # Fetch price data (portfolio + benchmark)
    all_tickers = list(set(tickers + [benchmark]))
    prices = fetch_prices(all_tickers, portfolio_input.start_date, portfolio_input.end_date)

    # Separate portfolio prices from benchmark
    port_tickers = [t for t in tickers if t in prices.columns]
    port_prices = prices[port_tickers]

    bm_prices = prices[[benchmark]] if benchmark in prices.columns else None

    # Run backtest
    portfolio_series, weight_history = run_backtest(
        port_prices,
        weights,
        portfolio_input.rebalance_frequency,
        portfolio_input.initial_capital,
    )

    portfolio_returns = portfolio_series.pct_change().dropna()

    # Benchmark series
    benchmark_series = None
    bm_returns = None
    if bm_prices is not None:
        bm_daily = bm_prices[benchmark].dropna()
        bm_aligned = bm_daily.reindex(portfolio_series.index).dropna()
        # Index benchmark to same initial capital
        bm_norm = bm_aligned / bm_aligned.iloc[0] * portfolio_input.initial_capital
        benchmark_series = bm_norm
        bm_returns = bm_aligned.pct_change().dropna()

    # Metrics
    raw_metrics = calculate_metrics(portfolio_returns, bm_returns, RISK_FREE_RATE)
    monthly_grid = build_monthly_returns_grid(portfolio_returns)

    # Build drawdown series
    cum = (1 + portfolio_returns).cumprod()
    cum_max = cum.cummax()
    dd = ((cum - cum_max) / cum_max).fillna(0)

    # Assemble result
    equity_curve = [
        TimeSeriesPoint(date=d.strftime("%Y-%m-%d"), value=round(v, 4))
        for d, v in portfolio_series.items()
    ]
    drawdown_series = [
        DrawdownPoint(date=d.strftime("%Y-%m-%d"), drawdown=round(v, 6))
        for d, v in dd.items()
    ]
    bm_curve = None
    if benchmark_series is not None:
        bm_curve = [
            TimeSeriesPoint(date=d.strftime("%Y-%m-%d"), value=round(v, 4))
            for d, v in benchmark_series.items()
        ]

    metrics = PortfolioMetrics(**raw_metrics)

    return BacktestResult(
        equity_curve=equity_curve,
        benchmark_curve=bm_curve,
        drawdown_series=drawdown_series,
        monthly_returns=MonthlyReturns(data=monthly_grid),
        metrics=metrics,
        weight_history=weight_history.to_dict(orient="records"),
    )
