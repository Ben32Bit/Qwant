import numpy as np
import pandas as pd
from typing import Optional
from app.utils.constants import TRADING_DAYS_PER_YEAR


def calculate_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.05,
) -> dict:
    """Calculate all portfolio performance metrics."""
    ann = TRADING_DAYS_PER_YEAR
    rfr_daily = (1 + risk_free_rate) ** (1 / ann) - 1

    # --- Portfolio curve ---
    if portfolio_returns.empty:
        raise ValueError("Portfolio returns series is empty — no tradeable date range found for these assets.")
    cum = (1 + portfolio_returns).cumprod()
    total_return = cum.iloc[-1] - 1
    n_days = len(portfolio_returns)
    cagr = (1 + total_return) ** (ann / n_days) - 1

    # Volatility
    vol = portfolio_returns.std() * np.sqrt(ann)

    # Sharpe
    excess = portfolio_returns - rfr_daily
    sharpe = (excess.mean() / portfolio_returns.std() * np.sqrt(ann)) if portfolio_returns.std() > 0 else 0.0

    # Sortino
    downside = portfolio_returns[portfolio_returns < rfr_daily]
    downside_dev = downside.std() * np.sqrt(ann) if len(downside) > 0 else 1e-10
    sortino = (cagr - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0

    # Max drawdown
    cum_max = cum.cummax()
    drawdown_series = (cum - cum_max) / cum_max
    max_dd = drawdown_series.min()

    # Max drawdown duration
    dd_duration = _max_drawdown_duration(drawdown_series)

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # VaR / CVaR
    var_95 = float(np.percentile(portfolio_returns, 5)) * np.sqrt(ann)
    cvar_95 = float(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()) * np.sqrt(ann)

    # Best / worst year and month
    monthly = _resample_returns(portfolio_returns, "ME")
    yearly = _resample_returns(portfolio_returns, "YE")

    best_year = _best_worst(yearly, best=True)
    worst_year = _best_worst(yearly, best=False)
    best_month = _best_worst(monthly, best=True)
    worst_month = _best_worst(monthly, best=False)

    metrics = {
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "volatility": round(vol, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "max_drawdown": round(max_dd, 6),
        "max_drawdown_duration_days": dd_duration,
        "downside_deviation": round(downside_dev, 6),
        "var_95": round(var_95, 6),
        "cvar_95": round(cvar_95, 6),
        "best_year": best_year,
        "worst_year": worst_year,
        "best_month": best_month,
        "worst_month": worst_month,
    }

    # --- Benchmark-relative ---
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        bm = benchmark_returns.reindex(portfolio_returns.index).dropna()
        port_aligned = portfolio_returns.reindex(bm.index).dropna()

        if len(bm) > 10:
            bm_var = bm.var()
            beta = float(np.cov(port_aligned, bm)[0, 1] / bm_var) if bm_var > 0 else 0.0
            bm_cum = (1 + bm).cumprod()
            bm_cagr = (bm_cum.iloc[-1]) ** (ann / len(bm)) - 1
            alpha = float(cagr - (risk_free_rate + beta * (bm_cagr - risk_free_rate)))
            corr = float(port_aligned.corr(bm))
            r_squared = corr ** 2
            active = port_aligned - bm
            tracking_error = float(active.std() * np.sqrt(ann))
            info_ratio = float(active.mean() / active.std() * np.sqrt(ann)) if active.std() > 0 else 0.0
            treynor = float((cagr - risk_free_rate) / beta) if beta != 0 else 0.0
            up_capture, down_capture = _capture_ratios(port_aligned, bm)

            metrics.update({
                "beta": round(beta, 4),
                "alpha": round(alpha, 6),
                "r_squared": round(r_squared, 4),
                "information_ratio": round(info_ratio, 4),
                "tracking_error": round(tracking_error, 6),
                "treynor": round(treynor, 4),
                "up_capture": round(up_capture, 4),
                "down_capture": round(down_capture, 4),
            })

    return metrics


def build_monthly_returns_grid(portfolio_returns: pd.Series) -> dict:
    """Build year × month grid of monthly returns."""
    monthly = _resample_returns(portfolio_returns, "ME")
    grid: dict[str, dict[str, float | None]] = {}
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for date, ret in monthly.items():
        year = str(date.year)
        month = month_names[date.month - 1]
        if year not in grid:
            grid[year] = {m: None for m in month_names}
        grid[year][month] = round(float(ret), 6)

    return grid


# --- Helpers ---

def _resample_returns(returns: pd.Series, freq: str) -> pd.Series:
    """Resample daily returns to period returns."""
    prices = (1 + returns).cumprod()
    period_prices = prices.resample(freq).last()
    return period_prices.pct_change().dropna()


def _best_worst(series: pd.Series, best: bool) -> dict | None:
    if series.empty:
        return None
    val = series.max() if best else series.min()
    date = series.idxmax() if best else series.idxmin()
    return {"period": str(date.date()), "return": round(float(val), 6)}


def _max_drawdown_duration(drawdown_series: pd.Series) -> int:
    """Find the longest number of days spent in drawdown."""
    in_dd = drawdown_series < 0
    max_dur = 0
    cur_dur = 0
    for v in in_dd:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    return max_dur


def _capture_ratios(port: pd.Series, bench: pd.Series) -> tuple[float, float]:
    up_mask = bench > 0
    down_mask = bench < 0

    def period_return(s: pd.Series) -> float:
        return float((1 + s).prod() ** (TRADING_DAYS_PER_YEAR / max(len(s), 1)) - 1)

    if up_mask.sum() > 0:
        up_cap = period_return(port[up_mask]) / period_return(bench[up_mask])
    else:
        up_cap = 0.0

    if down_mask.sum() > 0:
        down_cap = period_return(port[down_mask]) / period_return(bench[down_mask])
    else:
        down_cap = 0.0

    return up_cap, down_cap
