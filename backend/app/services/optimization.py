import numpy as np
import pandas as pd
from scipy.optimize import minimize
from app.utils.constants import TRADING_DAYS_PER_YEAR


def _cov_matrix(returns_df: pd.DataFrame) -> np.ndarray:
    return returns_df.cov().values * TRADING_DAYS_PER_YEAR


def equal_weight(tickers: list[str]) -> dict[str, float]:
    n = len(tickers)
    w = 1.0 / n
    return {t: w for t in tickers}


def min_variance(returns_df: pd.DataFrame, allow_short: bool = False) -> dict[str, float]:
    tickers = list(returns_df.columns)
    n = len(tickers)
    cov = _cov_matrix(returns_df)

    def objective(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(-1.0, 1.0) if allow_short else (0.0, 1.0)] * n
    x0 = np.array([1.0 / n] * n)

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return dict(zip(tickers, result.x))


def max_sharpe(returns_df: pd.DataFrame, risk_free_rate: float = 0.05, allow_short: bool = False) -> dict[str, float]:
    tickers = list(returns_df.columns)
    n = len(tickers)
    cov = _cov_matrix(returns_df)
    mean_returns = returns_df.mean().values * TRADING_DAYS_PER_YEAR

    def neg_sharpe(w):
        port_ret = w @ mean_returns
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-10:
            return 0.0
        return -(port_ret - risk_free_rate) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(-1.0, 1.0) if allow_short else (0.0, 1.0)] * n
    x0 = np.array([1.0 / n] * n)

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return dict(zip(tickers, result.x))


def risk_parity(returns_df: pd.DataFrame) -> dict[str, float]:
    """Equal risk contribution portfolio."""
    tickers = list(returns_df.columns)
    n = len(tickers)
    cov = _cov_matrix(returns_df)

    def risk_contribution(w):
        port_var = w @ cov @ w
        marginal_contrib = cov @ w
        risk_contrib = w * marginal_contrib / port_var
        return risk_contrib

    def objective(w):
        rc = risk_contribution(w)
        target = np.ones(n) / n
        return np.sum((rc - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.array([1.0 / n] * n)

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return dict(zip(tickers, result.x))


def inverse_vol(returns_df: pd.DataFrame) -> dict[str, float]:
    """Weight inversely proportional to historical volatility."""
    tickers = list(returns_df.columns)
    vols = returns_df.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    return dict(zip(tickers, weights.values))


def apply_strategy(
    strategy: str,
    returns_df: pd.DataFrame,
    current_weights: dict[str, float],
    risk_free_rate: float = 0.05,
) -> dict[str, float]:
    """Apply the given optimization strategy and return optimized weights."""
    if strategy == "custom":
        return current_weights
    elif strategy == "equal_weight":
        return equal_weight(list(returns_df.columns))
    elif strategy == "min_variance":
        return min_variance(returns_df)
    elif strategy == "max_sharpe":
        return max_sharpe(returns_df, risk_free_rate)
    elif strategy == "risk_parity":
        return risk_parity(returns_df)
    elif strategy == "inverse_vol":
        return inverse_vol(returns_df)
    else:
        return current_weights
