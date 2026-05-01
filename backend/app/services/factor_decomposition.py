"""
Factor Model Decomposition (FF5 + Momentum)
============================================
Regresses portfolio excess returns on:
  Mkt-RF, SMB, HML, RMW (quality/profitability), CMA, Mom (Momentum)

Factor data is fetched from Ken French's official data library via
pandas-datareader. Falls back to ETF proxies (yfinance) if unavailable.
Momentum (UMD) is included when the daily Ken French dataset is accessible;
ETF-proxy path uses 5-factor only.

References
----------
Fama, E.F. & French, K.R. (2015). A five-factor asset pricing model.
  Journal of Financial Economics, 116(1), 1–22.
  https://doi.org/10.1016/j.jfineco.2014.10.010

Carhart, M.M. (1997). On persistence in mutual fund performance.
  Journal of Finance, 52(1), 57–82.
  https://doi.org/10.1111/j.1540-6261.1997.tb03808.x

Novy-Marx, R. (2013). The other side of value: The gross profitability premium.
  Journal of Financial Economics, 108(1), 1–28.
  https://doi.org/10.1016/j.jfineco.2013.01.003
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import os
from typing import Optional

TRADING_DAYS = 252
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))


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


def _build_result(coeffs, t_stats, r_squared, n_obs,
                  factor_keys: list[str] | None = None) -> dict:
    if factor_keys is None:
        factor_keys = ["mkt_rf", "smb", "hml", "rmw", "cma"]

    result = {"r_squared": round(float(r_squared), 3), "n_obs": int(n_obs)}

    alpha_daily = float(coeffs[0])
    alpha_t     = float(t_stats[0])
    result["alpha"]        = round(float((1 + alpha_daily) ** TRADING_DAYS - 1), 4)
    result["alpha_t_stat"] = round(alpha_t, 2)
    result["alpha_stars"]  = _sig_stars(alpha_t)

    for i, name in enumerate(factor_keys):
        val = float(coeffs[i + 1])
        t   = float(t_stats[i + 1])
        result[name]             = round(val, 3)
        result[f"{name}_t_stat"] = round(t, 2)
        result[f"{name}_stars"]  = _sig_stars(t)

    result["factors_used"] = factor_keys
    return result


def _via_ken_french(returns: pd.Series) -> dict:
    import pandas_datareader.data as web

    start = returns.index[0].strftime("%Y-%m-%d")
    end   = returns.index[-1].strftime("%Y-%m-%d")

    ff5 = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3_daily",
        "famafrench",
        start=start,
        end=end,
    )[0] / 100

    mom_col = None
    try:
        mom_raw = web.DataReader(
            "F-F_Momentum_Factor_daily", "famafrench", start=start, end=end
        )[0] / 100
        mom_col = mom_raw["Mom"].rename("Mom")
    except Exception:
        pass

    df = pd.DataFrame({"R": returns}).join(ff5, how="inner")
    if mom_col is not None:
        df = df.join(mom_col, how="left")
    df = df.dropna(subset=["R", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])

    if len(df) < 60:
        raise ValueError("Insufficient overlapping observations")

    use_mom = (mom_col is not None
               and "Mom" in df.columns
               and df["Mom"].notna().mean() > 0.9)

    factor_french_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    factor_keys        = ["mkt_rf", "smb", "hml", "rmw", "cma"]
    if use_mom:
        df_mom = df.dropna(subset=["Mom"])
        factor_french_cols = factor_french_cols + ["Mom"]
        factor_keys        = factor_keys + ["mom"]
        df_reg = df_mom
    else:
        df_reg = df

    X = np.column_stack([np.ones(len(df_reg))]
                        + [df_reg[c].values for c in factor_french_cols])
    res = _ols(X, (df_reg["R"] - df_reg["RF"]).values)
    return _build_result(res["coeffs"], res["t_stats"],
                         res["r_squared"], res["n_obs"], factor_keys)


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
    cma      = -(px_ret["MTUM"] - px_ret["SPY"])

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
    result = _build_result(res["coeffs"], res["t_stats"], res["r_squared"], res["n_obs"],
                           ["mkt_rf", "smb", "hml", "rmw", "cma"])
    result["proxy_source"] = "etf_proxies"
    return result


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
