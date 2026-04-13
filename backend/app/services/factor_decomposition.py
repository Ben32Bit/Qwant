"""
Fama-French Five-Factor Decomposition
======================================
Regresses portfolio excess returns on the five Fama-French factors:
  Mkt-RF, SMB, HML, RMW, CMA

Factor data is fetched from Ken French's official data library via
pandas-datareader. Falls back to ETF proxies (yfinance) if unavailable.

Reference
---------
Fama, E.F. & French, K.R. (2015). A five-factor asset pricing model.
Journal of Financial Economics, 116(1), 1–22.
https://doi.org/10.1016/j.jfineco.2014.10.010
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import os
from typing import Optional

TRADING_DAYS = 252
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))

# ETF proxies used as fallback when Ken French data is unavailable
_ETF_PROXIES = {
    "market": "SPY",
    "smb":    ["IWM", "IWB"],   # Russell 2000 minus Russell 1000
    "hml":    ["IWD", "IWF"],   # Russell 1000 Value minus Growth
    "rmw":    ["QUAL", "USMV"],  # Quality minus Min-Vol (profitability proxy)
    "cma":    ["MTUM", "SPY"],   # Low investment proxy (inverse momentum)
}


def _ols(X: np.ndarray, y: np.ndarray) -> dict:
    """OLS with t-statistics and R²."""
    n, k = X.shape
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coeffs
    residuals = y - y_hat
    sigma2 = np.sum(residuals ** 2) / (n - k)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    t_stats = np.where(se > 0, coeffs / se, np.nan)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"coeffs": coeffs, "t_stats": t_stats, "r_squared": r2, "n_obs": n}


def _sig_stars(t: float) -> str:
    a = abs(t)
    if a > 3.29: return "***"
    if a > 2.58: return "**"
    if a > 1.96: return "*"
    return ""


def _build_result(coeffs, t_stats, r_squared, n_obs) -> dict:
    names = ["alpha", "mkt_rf", "smb", "hml", "rmw", "cma"]
    result = {"r_squared": round(float(r_squared), 3), "n_obs": int(n_obs)}
    for i, name in enumerate(names):
        val = float(coeffs[i])
        t   = float(t_stats[i])
        if name == "alpha":
            # Annualise daily alpha
            val_ann = float((1 + val) ** TRADING_DAYS - 1)
            result["alpha"]          = round(val_ann, 4)
            result["alpha_t_stat"]   = round(t, 2)
            result["alpha_stars"]    = _sig_stars(t)
        else:
            result[name]                  = round(val, 3)
            result[f"{name}_t_stat"]      = round(t, 2)
            result[f"{name}_stars"]       = _sig_stars(t)
    return result


# ── Primary: Ken French data library ─────────────────────────────────────────

def _via_ken_french(returns: pd.Series) -> dict:
    import pandas_datareader.data as web

    start = returns.index[0].strftime("%Y-%m-%d")
    end   = returns.index[-1].strftime("%Y-%m-%d")

    ff5 = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3_daily",
        "famafrench",
        start=start,
        end=end,
    )[0]
    ff5 = ff5 / 100  # percent → decimal

    df = pd.DataFrame({"R": returns}).join(ff5, how="inner").dropna()
    if len(df) < 60:
        raise ValueError("Insufficient overlapping observations")

    excess = df["R"] - df["RF"]
    X = np.column_stack([
        np.ones(len(df)),
        df["Mkt-RF"].values,
        df["SMB"].values,
        df["HML"].values,
        df["RMW"].values,
        df["CMA"].values,
    ])
    res = _ols(X, excess.values)
    return _build_result(res["coeffs"], res["t_stats"], res["r_squared"], res["n_obs"])


# ── Fallback: ETF proxies via yfinance ────────────────────────────────────────

def _via_etf_proxies(returns: pd.Series) -> dict:
    from app.services.data_service import fetch_prices

    start = returns.index[0].strftime("%Y-%m-%d")
    end   = returns.index[-1].strftime("%Y-%m-%d")

    tickers = ["SPY", "IWM", "IWB", "IWD", "IWF", "QUAL", "USMV", "MTUM"]
    prices  = fetch_prices(tickers, start, end)
    px_ret  = prices.pct_change().dropna()

    rf_daily = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    mkt_rf   = px_ret["SPY"] - rf_daily
    smb      = px_ret["IWM"]  - px_ret["IWB"]
    hml      = px_ret["IWD"]  - px_ret["IWF"]
    rmw      = px_ret["QUAL"] - px_ret["USMV"]
    cma      = -(px_ret["MTUM"] - px_ret["SPY"])  # high momentum ≈ aggressive investors

    factors = pd.DataFrame({"mkt_rf": mkt_rf, "smb": smb, "hml": hml, "rmw": rmw, "cma": cma})
    df = pd.DataFrame({"R": returns}).join(factors, how="inner").dropna()
    if len(df) < 60:
        raise ValueError("Insufficient overlapping observations")

    excess = df["R"] - rf_daily
    X = np.column_stack([
        np.ones(len(df)),
        df["mkt_rf"].values,
        df["smb"].values,
        df["hml"].values,
        df["rmw"].values,
        df["cma"].values,
    ])
    res = _ols(X, excess.values)
    result = _build_result(res["coeffs"], res["t_stats"], res["r_squared"], res["n_obs"])
    result["proxy_source"] = "etf_proxies"
    return result


# ── Public entry point ────────────────────────────────────────────────────────

def compute_ff5(portfolio_returns: pd.Series) -> Optional[dict]:
    """
    Compute Fama-French 5-factor decomposition for the given daily return series.
    Returns None if computation fails (e.g. too few observations).
    """
    if portfolio_returns is None or len(portfolio_returns) < 60:
        return None
    try:
        result = _via_ken_french(portfolio_returns)
        result["source"] = "ken_french"
        return result
    except Exception as e1:
        try:
            result = _via_etf_proxies(portfolio_returns)
            result["source"] = "etf_proxies"
            return result
        except Exception as e2:
            print(f"FF5 decomposition failed — KF: {e1} | ETF: {e2}")
            return None
