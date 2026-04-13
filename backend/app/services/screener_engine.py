import math
import pandas as pd
import numpy as np
import logging
from app.services.data_service import fetch_prices
from app.services.metrics import calculate_metrics, build_monthly_returns_grid
from app.services.backtest_engine import _thin
from app.models.screener import (
    ScreenerResult, ScreenerWindow, TickerWindowResult,
    ScreenRequest, RotationBacktestRequest,
)
from app.models.backtest_result import (
    BacktestResult, TimeSeriesPoint, DrawdownPoint,
    MonthlyReturns, PortfolioMetrics,
)
from app.utils.constants import TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.05

METRIC_LABELS = {
    "return":       "Total Return",
    "sharpe":       "Sharpe Ratio",
    "volatility":   "Volatility (lower=better)",
    "max_drawdown": "Max Drawdown (lower=better)",
    "momentum_3m":  "3-Month Momentum",
}

FREQ_MAP = {
    "weekly":    "W",
    "monthly":   "ME",
    "quarterly": "QE",
    "annually":  "YE",
}


def _generate_windows(index: pd.DatetimeIndex, freq: str) -> list[tuple]:
    """
    Split a trading-day index into non-overlapping calendar windows.
    Returns list of (window_start_date, window_end_date) as Timestamps.
    """
    pd_freq = FREQ_MAP.get(freq, "QE")
    period_ends = pd.date_range(start=index[0], end=index[-1], freq=pd_freq)

    windows = []
    prev = index[0]
    for pe in period_ends:
        # All trading days in [prev, pe]
        mask = (index >= prev) & (index <= pe)
        chunk = index[mask]
        if len(chunk) < 5:
            continue
        windows.append((chunk[0], chunk[-1]))
        # Next window starts on the first trading day after pe
        after = index[index > pe]
        prev = after[0] if len(after) > 0 else pe

    # Catch any remaining days after the last period-end
    if len(windows) > 0:
        last_end = windows[-1][1]
        remaining = index[index > last_end]
        if len(remaining) >= 5:
            windows.append((remaining[0], remaining[-1]))

    return windows


def _window_label(start: pd.Timestamp, freq: str) -> str:
    if freq == "quarterly":
        q = (start.month - 1) // 3 + 1
        return f"{start.year} Q{q}"
    if freq == "monthly":
        return start.strftime("%b %Y")
    if freq == "annually":
        return str(start.year)
    return start.strftime("%Y-%m-%d")


def _score_ticker(returns: pd.Series, metric: str) -> float:
    """Compute the screening metric for one ticker's return series."""
    if returns.empty or len(returns) < 3:
        return float("-inf")

    if metric == "return":
        return float((1 + returns).prod() - 1)

    if metric in ("momentum_3m", "momentum"):
        return float((1 + returns).prod() - 1)

    if metric == "sharpe":
        std = returns.std()
        if std == 0:
            return 0.0
        return float(returns.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))

    if metric == "volatility":
        # Negate so lower vol → higher score (better rank)
        return -float(returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR))

    if metric == "max_drawdown":
        cum = (1 + returns).cumprod()
        dd = ((cum - cum.cummax()) / cum.cummax()).min()
        # Negate so smaller drawdown → higher score
        return -float(dd)

    return float("-inf")


def run_screener(req: ScreenRequest) -> ScreenerResult:
    prices = fetch_prices(req.tickers, req.start_date, req.end_date)
    valid = [t for t in req.tickers if t in prices.columns]
    if not valid:
        raise ValueError("No valid tickers found for the requested period.")

    returns = prices[valid].pct_change().dropna()
    windows = _generate_windows(returns.index, req.window_freq)

    result_windows: list[ScreenerWindow] = []
    for w_start, w_end in windows:
        wr = returns.loc[w_start:w_end]
        if wr.empty:
            continue

        scores = {
            t: _score_ticker(wr[t].dropna(), req.metric)
            for t in valid
            if t in wr.columns
        }
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Normalise metric_value: for display show the raw financial value
        # For inverted metrics (vol, DD) we un-negate for display
        def display_val(ticker: str, raw_score: float) -> float:
            if req.metric == "volatility":
                return round(-raw_score, 4)   # positive vol
            if req.metric == "max_drawdown":
                return round(-raw_score, 4)   # positive DD magnitude
            return round(raw_score, 4)

        rankings = [
            TickerWindowResult(
                ticker=t,
                metric_value=display_val(t, v),
                rank=i + 1,
            )
            for i, (t, v) in enumerate(sorted_scores)
        ]

        result_windows.append(ScreenerWindow(
            window_start=w_start.strftime("%Y-%m-%d"),
            window_end=w_end.strftime("%Y-%m-%d"),
            label=_window_label(w_start, req.window_freq),
            rankings=rankings,
            winner=sorted_scores[0][0] if sorted_scores else None,
        ))

    return ScreenerResult(
        windows=result_windows,
        metric=req.metric,
        metric_label=METRIC_LABELS.get(req.metric, req.metric),
        window_freq=req.window_freq,
        tickers=valid,
        top_n=req.top_n,
        screen_description=req.screen_description,
    )


def run_rotation_backtest(req: RotationBacktestRequest) -> BacktestResult:
    """
    Backtest a momentum rotation strategy derived from screener results.

    Strategy: at the start of each window, hold the top-N assets from the
    PREVIOUS window in equal weight (no lookahead bias).
    """
    screen = req.screen_result
    benchmark = req.benchmark
    initial_capital = req.initial_capital
    top_n_held = max(1, req.top_n_held)

    if len(screen.windows) < 2:
        raise ValueError("Need at least 2 windows to build a rotation strategy.")

    # Fetch prices for all tickers + benchmark
    all_tickers = list(set(screen.tickers + [benchmark]))
    start = screen.windows[0].window_start
    end = screen.windows[-1].window_end
    prices = fetch_prices(all_tickers, start, end)

    valid = [t for t in screen.tickers if t in prices.columns]
    returns_df = prices[valid + ([benchmark] if benchmark in prices.columns else [])].pct_change().dropna()

    # Build rotation schedule (using previous window's rankings — no lookahead)
    # rotation_map: {window_start_str → [top_n tickers]}
    rotation_map: dict[str, list[str]] = {}
    for i in range(1, len(screen.windows)):
        prev_window = screen.windows[i - 1]
        curr_window = screen.windows[i]
        top_tickers = [
            r.ticker for r in sorted(prev_window.rankings, key=lambda x: x.rank)
            if r.ticker in valid
        ][:top_n_held]
        if top_tickers:
            rotation_map[curr_window.window_start] = top_tickers

    # Initial holding: top asset from first window
    first_tops = [
        r.ticker for r in sorted(screen.windows[0].rankings, key=lambda x: x.rank)
        if r.ticker in valid
    ][:top_n_held]
    current_holdings = first_tops if first_tops else valid[:top_n_held]

    # Build holding_schedule for frontend visualisation
    # Maps each window to the tickers actually held during that window
    holding_schedule = []
    for i, win in enumerate(screen.windows):
        if i == 0:
            held = first_tops if first_tops else valid[:top_n_held]
        else:
            held = rotation_map.get(win.window_start, holding_schedule[-1]["tickers"] if holding_schedule else valid[:top_n_held])
        holding_schedule.append({
            "window_start": win.window_start,
            "window_end": win.window_end,
            "label": win.label,
            "tickers": held,
        })

    # Run daily backtest
    portfolio_values = [initial_capital]
    rotation_date_strs = list(rotation_map.keys())
    rotation_date_set = set(rotation_date_strs)

    for date in returns_df.index:
        date_str = date.strftime("%Y-%m-%d")
        if date_str in rotation_date_set:
            current_holdings = rotation_map[date_str]

        # Equal-weight the current holdings
        day_ret = 0.0
        held = [t for t in current_holdings if t in returns_df.columns]
        if held:
            w = 1.0 / len(held)
            day_ret = sum(
                w * float(returns_df.loc[date, t]) for t in held
            )
        portfolio_values.append(portfolio_values[-1] * (1 + day_ret))

    portfolio_series = pd.Series(portfolio_values[1:], index=returns_df.index)
    portfolio_returns = portfolio_series.pct_change().dropna()

    # Benchmark curve
    bm_series = None
    bm_returns = None
    if benchmark in prices.columns:
        bm_raw = prices[benchmark].dropna()
        bm_aligned = bm_raw.reindex(portfolio_series.index).dropna()
        if not bm_aligned.empty:
            bm_series = bm_aligned / bm_aligned.iloc[0] * initial_capital
            bm_returns = bm_aligned.pct_change().dropna()

    # Metrics + series
    from app.services.factor_decomposition import compute_ff5
    raw_metrics = calculate_metrics(portfolio_returns, bm_returns, RISK_FREE_RATE)
    monthly_grid = build_monthly_returns_grid(portfolio_returns)
    ff5 = compute_ff5(portfolio_returns)

    cum = (1 + portfolio_returns).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).fillna(0)

    def thin_ts(series: pd.Series) -> list[TimeSeriesPoint]:
        s = _thin(series)
        return [
            TimeSeriesPoint(date=d.strftime("%Y-%m-%d"), value=round(float(v), 4))
            for d, v in s.items()
        ]

    return BacktestResult(
        equity_curve=thin_ts(portfolio_series),
        benchmark_curve=thin_ts(bm_series) if bm_series is not None else None,
        drawdown_series=[
            DrawdownPoint(date=d.strftime("%Y-%m-%d"), drawdown=round(float(v), 6))
            for d, v in _thin(dd).items()
        ],
        monthly_returns=MonthlyReturns(data=monthly_grid),
        metrics=PortfolioMetrics(**raw_metrics),
        rebalance_dates=rotation_date_strs,
        holding_schedule=holding_schedule,
        ff5_decomposition=ff5,
    )
