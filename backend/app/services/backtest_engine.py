import pandas as pd
import numpy as np
from typing import Optional
from app.services.data_service import fetch_prices
from app.services.metrics import calculate_metrics, build_monthly_returns_grid
from app.services.factor_decomposition import compute_ff5
from app.models.portfolio import PortfolioInput
from app.models.backtest_result import (
    BacktestResult, TimeSeriesPoint, DrawdownPoint,
    MonthlyReturns, PortfolioMetrics, RollingMetrics,
)
from app.utils.constants import TRADING_DAYS_PER_YEAR, FX_PAIRS
import logging
import os

logger = logging.getLogger(__name__)

RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))
MAX_CHART_POINTS = 500  # thin series for frontend performance


def _thin(series: pd.Series) -> pd.Series:
    """Downsample a series to at most MAX_CHART_POINTS, always keeping the last point."""
    if len(series) <= MAX_CHART_POINTS:
        return series
    step = len(series) // MAX_CHART_POINTS
    idx = list(range(0, len(series), step))
    if idx[-1] != len(series) - 1:
        idx.append(len(series) - 1)
    return series.iloc[idx]


def _get_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set:
    if freq == "none":
        return {index[0]}
    if freq == "daily":
        return set(index)

    freq_map = {"weekly": "W", "monthly": "ME", "quarterly": "QE", "annually": "YE"}
    period_freq = freq_map.get(freq, "QE")
    rebalance = pd.date_range(start=index[0], end=index[-1], freq=period_freq)
    result = set()
    for rd in rebalance:
        candidates = index[index >= rd]
        if len(candidates) > 0:
            result.add(candidates[0])
    result.add(index[0])
    return result


def run_backtest(
    prices_df: pd.DataFrame,
    weights: dict[str, float],
    rebalance_freq: str,
    initial_capital: float = 10_000.0,
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    tickers = [t for t in weights if t in prices_df.columns]
    if not tickers:
        raise ValueError("None of the requested tickers have price data available.")

    prices = prices_df[tickers].copy()

    # Find the latest start date across all tickers so we don't lose data to NaN propagation.
    # This handles momentum stocks with shorter history than the backtest window.
    first_valid = prices.apply(lambda col: col.first_valid_index())
    effective_start = first_valid.max()  # latest first-valid date across all tickers
    if effective_start is not None:
        prices = prices.loc[effective_start:]

    daily_returns = prices.pct_change().dropna()

    if daily_returns.empty:
        raise ValueError(
            f"No overlapping trading days found for tickers {tickers} in the requested date range. "
            "Some assets may not have data for the full period — try a shorter date range or different tickers."
        )

    target_weights = {t: float(weights[t]) for t in tickers}
    rebalance_dates = _get_rebalance_dates(daily_returns.index, rebalance_freq)

    # Track position values in dollar terms (positive = long, negative = short).
    # This correctly handles long/short and dollar-neutral portfolios.
    # The old approach — normalising by weight_sum — fails when weight_sum ≈ 0
    # (dollar-neutral pair): division by near-zero explodes weights to ±millions,
    # corrupting both the equity curve and the weight drift chart.
    portfolio_value = initial_capital
    position_dollars = {t: portfolio_value * target_weights[t] for t in tickers}
    portfolio_values = [initial_capital]
    weight_records = []
    rebalance_date_strs = []

    for date in daily_returns.index:
        is_rebalance = date in rebalance_dates
        if is_rebalance:
            # Reset position sizes to target weights × current portfolio NAV
            for t in tickers:
                position_dollars[t] = portfolio_value * target_weights[t]
            rebalance_date_strs.append(date.strftime("%Y-%m-%d"))

        day_rets = daily_returns.loc[date]

        # Portfolio P&L = Σ (position_dollar × asset_return)
        daily_pnl = sum(position_dollars[t] * float(day_rets[t]) for t in tickers)
        new_portfolio_value = portfolio_value + daily_pnl
        portfolio_values.append(new_portfolio_value)

        # Update individual position values
        for t in tickers:
            position_dollars[t] *= (1 + float(day_rets[t]))

        portfolio_value = new_portfolio_value

        # Weights for drift chart = position_dollar / portfolio_NAV
        # For long-only portfolios this equals the familiar drifted weight.
        # For dollar-neutral pairs it gives sensible ±1.0x values, not ±100x.
        if abs(portfolio_value) > 1e-10:
            w_now = {t: position_dollars[t] / portfolio_value for t in tickers}
        else:
            w_now = dict(target_weights)

        weight_records.append({"date": date.strftime("%Y-%m-%d"), **w_now})

    portfolio_series = pd.Series(portfolio_values[1:], index=daily_returns.index)
    weight_history = pd.DataFrame(weight_records)
    return portfolio_series, weight_history, rebalance_date_strs


def _compute_rolling_metrics(
    portfolio_returns: pd.Series,
    bm_returns: Optional[pd.Series] = None,
) -> RollingMetrics:
    """Compute rolling Sharpe (252d), volatility (60d), and beta (126d)."""
    rfr_daily = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    # Rolling Sharpe
    excess = portfolio_returns - rfr_daily
    roll_sharpe = (
        excess.rolling(252).mean() / portfolio_returns.rolling(252).std()
    ) * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Rolling volatility (annualised)
    roll_vol = portfolio_returns.rolling(60).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    def to_pts(s: pd.Series) -> list[TimeSeriesPoint]:
        s = s.dropna()
        s = _thin(s)
        return [TimeSeriesPoint(date=d.strftime("%Y-%m-%d"), value=round(float(v), 4)) for d, v in s.items()]

    rolling = RollingMetrics(
        sharpe=to_pts(roll_sharpe),
        volatility=to_pts(roll_vol),
    )

    if bm_returns is not None:
        bm_aligned = bm_returns.reindex(portfolio_returns.index).dropna()
        port_aligned = portfolio_returns.reindex(bm_aligned.index)
        roll_cov = port_aligned.rolling(126).cov(bm_aligned)
        roll_var = bm_aligned.rolling(126).var()
        roll_beta = (roll_cov / roll_var).replace([np.inf, -np.inf], np.nan)
        rolling.beta = to_pts(roll_beta)

    return rolling


def run_full_backtest(portfolio_input: PortfolioInput) -> BacktestResult:
    tickers = [a.ticker for a in portfolio_input.assets]
    weights = {a.ticker: a.weight for a in portfolio_input.assets}
    benchmark = portfolio_input.benchmark

    all_tickers = list(set(tickers + [benchmark]))
    prices = fetch_prices(all_tickers, portfolio_input.start_date, portfolio_input.end_date)

    port_tickers = [t for t in tickers if t in prices.columns]
    port_prices = prices[port_tickers]

    # Run backtest
    portfolio_series, weight_history, rebalance_dates_list = run_backtest(
        port_prices, weights, portfolio_input.rebalance_frequency, portfolio_input.initial_capital,
    )
    portfolio_returns = portfolio_series.pct_change().dropna()

    # Benchmark
    benchmark_series = None
    bm_returns = None
    if benchmark in prices.columns:
        bm_daily = prices[benchmark].dropna()
        bm_aligned = bm_daily.reindex(portfolio_series.index).dropna()
        if not bm_aligned.empty:
            benchmark_series = bm_aligned / bm_aligned.iloc[0] * portfolio_input.initial_capital
            bm_returns = bm_aligned.pct_change().dropna()

    # Metrics
    raw_metrics = calculate_metrics(portfolio_returns, bm_returns, RISK_FREE_RATE)
    monthly_grid = build_monthly_returns_grid(portfolio_returns)

    # Drawdown
    cum = (1 + portfolio_returns).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).fillna(0)

    # Rolling metrics
    rolling = _compute_rolling_metrics(portfolio_returns, bm_returns)

    # Asset correlation matrix (portfolio assets only)
    port_returns_df = port_prices.pct_change().dropna()
    corr_matrix = None
    if len(port_tickers) > 1:
        corr_matrix = port_returns_df[port_tickers].corr().round(3).to_dict()

    # Build response objects
    def thin_ts(series: pd.Series) -> list[TimeSeriesPoint]:
        s = _thin(series)
        return [TimeSeriesPoint(date=d.strftime("%Y-%m-%d"), value=round(float(v), 4)) for d, v in s.items()]

    equity_curve = thin_ts(portfolio_series)
    drawdown_series = [
        DrawdownPoint(date=d.strftime("%Y-%m-%d"), drawdown=round(float(v), 6))
        for d, v in _thin(dd).items()
    ]
    bm_curve = thin_ts(benchmark_series) if benchmark_series is not None else None

    # Thin weight history to ≤300 points (preserving first, last, and all rebalance dates)
    wh_step = max(1, len(weight_history) // 300)
    rebalance_date_set = set(rebalance_dates_list)
    rebalance_indices = [
        i for i, r in enumerate(weight_history.itertuples())
        if r.date in rebalance_date_set
    ]
    wh_thinned = weight_history.iloc[
        sorted(set(
            list(range(0, len(weight_history), wh_step))
            + rebalance_indices
            + [len(weight_history) - 1]
        ))
    ]
    # Round weights for compactness
    wh_records = []
    for _, row in wh_thinned.iterrows():
        rec = {"date": row["date"]}
        for t in port_tickers:
            if t in row:
                rec[t] = round(float(row[t]), 4)
        wh_records.append(rec)

    # FX-adjusted equity curves
    fx_curves = _compute_fx_curves(portfolio_series, portfolio_input.start_date, portfolio_input.end_date)

    # Fama-French Five-Factor decomposition
    ff5 = compute_ff5(portfolio_returns)

    return BacktestResult(
        equity_curve=equity_curve,
        benchmark_curve=bm_curve,
        drawdown_series=drawdown_series,
        monthly_returns=MonthlyReturns(data=monthly_grid),
        metrics=PortfolioMetrics(**raw_metrics),
        weight_history=wh_records,
        rebalance_dates=rebalance_dates_list,
        rolling_metrics=rolling,
        correlation_matrix=corr_matrix,
        fx_curves=fx_curves,
        ff5_decomposition=ff5,
    )


def _compute_fx_curves(
    portfolio_series: pd.Series,
    start_date: str,
    end_date: str,
) -> dict[str, list[TimeSeriesPoint]]:
    """Return FX-adjusted equity curves for each major currency pair."""
    import yfinance as yf

    fx_curves: dict[str, list[TimeSeriesPoint]] = {}

    for currency, (ticker, invert) in FX_PAIRS.items():
        try:
            raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if raw.empty:
                continue

            close = raw["Close"]
            if hasattr(close, "squeeze"):
                close = close.squeeze()
            close = close.dropna()

            # Align FX to portfolio dates (forward-fill weekends/holidays)
            fx_aligned = close.reindex(portfolio_series.index, method="ffill").dropna()
            if fx_aligned.empty or fx_aligned.iloc[0] == 0:
                continue

            # Normalise to "foreign currency per 1 USD"
            rate = (1.0 / fx_aligned) if invert else fx_aligned

            # Adjust portfolio: FCY value = USD_portfolio × (rate_t / rate_0)
            fx_factor = rate / rate.iloc[0]
            adjusted = portfolio_series.reindex(rate.index) * fx_factor
            adjusted = adjusted.dropna()

            thinned = _thin(adjusted)
            fx_curves[currency] = [
                TimeSeriesPoint(date=d.strftime("%Y-%m-%d"), value=round(float(v), 4))
                for d, v in thinned.items()
            ]
        except Exception as exc:
            logger.warning("FX curve failed for %s (%s): %s", currency, ticker, exc)

    return fx_curves
