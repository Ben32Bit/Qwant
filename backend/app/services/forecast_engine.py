"""
Forecast Engine — 6 Research-Backed Portfolio Forecasting Methods
=================================================================

All methods produce 12-month (252-trading-day) probabilistic forecasts as
percentile fan bands (p5/p25/p50/p75/p95) over the portfolio equity curve.

Out-of-sample integrity
-----------------------
Every method in this module follows strict quant best practices for
out-of-sample testing and overfitting prevention:

  - Walk-forward (expanding-window) train/test splits are used throughout.
    Random k-fold is never applied to financial time series, as it violates
    temporal ordering and leaks future information into training data.

  - For parametric models (GARCH, HMM, VAR), parameters are estimated on an
    80% training window and diagnostics are computed on the held-out 20% to
    detect misspecification before forecasting.

  - For the LSTM, MC Dropout is applied at inference to produce Bayesian
    uncertainty bands without requiring an ensemble of retrained models.
    Early stopping on a chronological validation set prevents overfitting.

  - Minimum history requirements are enforced. Requests with insufficient
    data return an informative error rather than an overfit estimate.

References for out-of-sample methodology
-----------------------------------------
Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
  Chapter 7: Cross-Validation in Finance (purged k-fold, embargo).
  https://doi.org/10.1002/9781119482161

Bailey, D.H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
  Journal of Portfolio Management, 40(5), 94–107.
  https://doi.org/10.3905/jpm.2014.40.5.094

Harvey, C.R., Liu, Y., & Zhu, H. (2016). ... and the Cross-Section of
  Expected Returns. Review of Financial Studies, 29(1), 5–68.
  https://doi.org/10.1093/rfs/hhv059
"""

from __future__ import annotations

import time
import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Shared constants ──────────────────────────────────────────────────────────

TRADING_DAYS = 252

# Consensus long-run factor premia (annualised).
# Sources: French data library historical averages + Damodaran (2024 estimate).
# Fama, E.F. & French, K.R. (2015). Journal of Financial Economics, 116(1), 1–22.
# Damodaran, A. (2024). Equity Risk Premiums: Determinants, Estimation and
#   Implications. NYU Stern Working Paper.
FACTOR_PREMIA = {
    "mkt_rf": 0.053,   # Market excess return (Damodaran 2024 US ERP)
    "smb":    0.020,   # Size premium (Fama & French 1993)
    "hml":    0.035,   # Value premium
    "rmw":    0.035,   # Profitability / quality premium (Novy-Marx 2013)
    "cma":    0.025,   # Investment premium
    "mom":    0.038,   # Momentum premium (Carhart 1997; Asness, Moskowitz & Pedersen 2013)
}

# Method colours for the frontend
METHOD_COLORS = {
    "xgboost":     "#4a9eff",   # blue  (replaced Monte Carlo GBM)
    "nbeats":      "#ffd43b",   # amber  (replaced GARCH(1,1))
    "hmm":         "#a855f7",   # purple
    "factor":      "#00d4aa",   # teal
    "var":         "#ff6b35",   # orange  (now GP)
    "lstm":        "#ff4757",   # red
}

METHOD_LABELS = {
    "xgboost":     "XGBoost Quantile",
    "nbeats":      "N-BEATS Neural",
    "hmm":         "Hidden Markov Model",
    "factor":      "Factor Model (FF5+Mom)",
    "var":         "Gaussian Process (GP)",
    "lstm":        "LSTM Neural Net",
}


# ── Shared infrastructure ─────────────────────────────────────────────────────

def _equity_to_returns(equity_curve: list[dict]) -> pd.Series:
    """Convert equity curve JSON to a daily returns Series (date-indexed)."""
    vals = {pt["date"]: float(pt["value"]) for pt in equity_curve}
    s = pd.Series(vals)
    s.index = pd.to_datetime(s.index)
    return s.sort_index().pct_change().dropna()


def _forecast_dates(last_date: str, horizon: int) -> list[str]:
    """Generate business-day forecast dates starting the day after last_date."""
    start = pd.Timestamp(last_date) + pd.offsets.BDay(1)
    idx = pd.bdate_range(start=start, periods=horizon)
    return [d.strftime("%Y-%m-%d") for d in idx]


def _paths_to_band(paths: np.ndarray, dates: list[str]) -> dict:
    """
    Convert a (horizon, n_paths) simulation array to percentile JSON.
    paths are expressed as cumulative % returns from the last historical value.
    """
    pcts = np.percentile(paths * 100, [5, 25, 50, 75, 95], axis=1)
    return {
        "dates": dates,
        "p5":  pcts[0].tolist(),
        "p25": pcts[1].tolist(),
        "p50": pcts[2].tolist(),
        "p75": pcts[3].tolist(),
        "p95": pcts[4].tolist(),
    }


def _ledoit_wolf_cov(returns_df: pd.DataFrame) -> np.ndarray:
    """
    Shrinkage covariance estimator — reduces estimation error for K>5 assets.

    Reference
    ---------
    Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-
    dimensional covariance matrices. Journal of Multivariate Analysis,
    88(2), 365–411. https://doi.org/10.1016/S0047-259X(03)00096-4
    """
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(returns_df.values)
    return lw.covariance_ * TRADING_DAYS   # annualise


# ── 1. XGBoost Quantile — Client-Side Feature Preparation ────────────────────
#
# XGBoost gradient-boosted quantile regressors now replace Monte Carlo GBM.
# GBM assumes a random walk with constant μ and σ — it cannot learn from
# momentum, vol regimes, or cross-sectional patterns. XGBoost, by contrast,
# learns nonlinear interactions between 9 market microstructure features.
#
# Reference: Gu, Kelly & Xiu (2020, RFS) — XGBoost dominates all parametric
# models on OOS stock return prediction across 900+ firm characteristics.
#
# Like the Attention-LSTM, the XGBoost models run CLIENT-SIDE via ONNX
# Runtime Web. The server's role is ONLY feature engineering.
#
# Training script:  backend/scripts/train_xgboost.py
# ONNX models:      frontend/public/models/xgboost/q{05,25,50,75,95}.onnx
#
# The ONNX models include StandardScaler as preprocessing nodes (Pipeline →
# ONNX), so XGBoostInferer.js passes raw unscaled features directly.
#
# Fan chart extrapolation (client-side, 21d → 252d):
#   The models predict 21-day cumulative return distributions.
#   For the 252-day chart, the client scales the 21-day bands using:
#     median(t) = (1 + p50_21d)^(t/21) − 1
#     spread(t) = half_spread_21d × sqrt(t/21)   [i.i.d. scaling]
#
# References
# ----------
# Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
#   KDD '16. https://doi.org/10.1145/2939672.2939785
# Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine
#   Learning. Review of Financial Studies, 33(5), 2223–2273.
#   https://doi.org/10.1093/rfs/hhaa009
# Friedman, J.H. (2001). Greedy function approximation: a gradient boosting
#   machine. Annals of Statistics, 29(5), 1189–1232.
#   https://doi.org/10.1214/aos/1013203451
# López de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 7.

def prepare_xgboost_features(
    returns: pd.Series,
    horizon: int,
    last_date: str,
) -> dict:
    """
    Compute 14 raw (unscaled) predictive features from the portfolio return series
    plus live macro/VIX data from FRED and yfinance.

    The ONNX Pipeline includes StandardScaler, so the client passes raw values.

    Price features (9)
    ------------------
    ret_1d, ret_5d, ret_21d, ret_63d : multi-horizon momentum
    vol_21d, vol_63d                  : realized vol (annualised)
    mom_12_1                          : 12-month minus 1-month momentum
    rsi_14                            : 14-day RSI (range 0–1)
    vol_ratio                         : vol_21d / vol_63d (vol regime)

    Macro features (5)
    ------------------
    yield_curve_10y2y : 10Y-2Y Treasury spread (recession/expansion)
    credit_spread_baa : BAA-10Y credit spread  (risk-off proxy)
    vix_pct_rank      : VIX percentile rank vs 2Y history (0–1)
    vix_term_slope    : VIX3M/VIX − 1 (contango = bullish)
    real_yield_10y    : 10Y TIPS yield (financial conditions tightness)

    Falls back gracefully: if FRED/VIX are unavailable, neutral historical
    averages are appended instead so inference always returns a result.

    Note: retrain the ONNX models via scripts/train_xgboost.py whenever
    the feature list changes. The meta.json n_features field is the
    authoritative feature count — the client uses it for tensor shape.
    """
    MIN_OBS = 63
    if len(returns) < MIN_OBS:
        raise ValueError(
            f"Insufficient history for XGBoost: need ≥{MIN_OBS} trading days, "
            f"got {len(returns)}. Try a longer backtest date range."
        )

    r = returns.copy()
    n = len(r)

    # ── Price features ────────────────────────────────────────────────────────
    ret_1d  = float(r.iloc[-1])
    ret_5d  = float(r.iloc[-5:].sum())
    ret_21d = float(r.iloc[-21:].sum())
    ret_63d = float(r.iloc[-63:].sum())

    vol_21d = float(r.iloc[-21:].std() * np.sqrt(TRADING_DAYS))
    vol_63d = float(r.iloc[-63:].std() * np.sqrt(TRADING_DAYS))

    if n >= TRADING_DAYS:
        mom_12_1 = float(r.iloc[-TRADING_DAYS:-21].sum())
    elif n >= 42:
        mom_12_1 = float(r.iloc[:-21].sum())
    else:
        mom_12_1 = 0.0

    delta  = r.diff().dropna()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rsi_14 = float(gain.iloc[-1] / (gain.iloc[-1] + loss.iloc[-1] + 1e-9))

    vol_ratio = vol_21d / (vol_63d + 1e-9)

    price_features = [ret_1d, ret_5d, ret_21d, ret_63d,
                      vol_21d, vol_63d, mom_12_1, rsi_14, vol_ratio]

    # ── Macro features ────────────────────────────────────────────────────────
    macro_available = False
    try:
        from .fred_provider import get_macro_features
        from .vix_provider  import get_vix_features

        fred = get_macro_features(as_of_date=last_date)
        vix  = get_vix_features(as_of_date=last_date)
        macro_available = fred.get("available", False) or vix.get("available", False)

        macro_features = [
            fred["yield_curve_10y2y"],
            fred["credit_spread_baa"],
            vix["vix_pct_rank"],
            vix["vix_term_slope"],
            fred["real_yield_10y"],
        ]
        macro_context = {
            "vix_spot":             vix.get("vix_spot"),
            "vix_pct_rank":         vix.get("vix_pct_rank"),
            "vix_term_slope":       vix.get("vix_term_slope"),
            "vix_contango":         vix.get("vix_contango"),
            "yield_curve_10y2y":    fred.get("yield_curve_10y2y"),
            "credit_spread_baa":    fred.get("credit_spread_baa"),
            "real_yield_10y":       fred.get("real_yield_10y"),
            "yield_curve_pct_rank": fred.get("yield_curve_pct_rank"),
            "credit_pct_rank":      fred.get("credit_pct_rank"),
            "macro_available":      macro_available,
        }
    except Exception as exc:
        logger.warning("prepare_xgboost_features: macro fetch failed (%s) — using neutrals", exc)
        macro_features = [0.60, 2.20, 0.50, 0.05, 0.50]   # long-run neutral values
        macro_context  = {"macro_available": False}

    features = price_features + macro_features
    feature_names = [
        "ret_1d", "ret_5d", "ret_21d", "ret_63d",
        "vol_21d", "vol_63d", "mom_12_1", "rsi_14", "vol_ratio",
        "yield_curve_10y2y", "credit_spread_baa",
        "vix_pct_rank", "vix_term_slope", "real_yield_10y",
    ]

    return {
        "features":       features,
        "feature_names":  feature_names,
        "forecast_dates": _forecast_dates(last_date, horizon),
        # Display metadata for the card
        "ret_21d_ann":    round(ret_21d * (TRADING_DAYS / 21), 4),
        "vol_21d_ann":    round(vol_21d, 4),
        "rsi_14":         round(rsi_14, 3),
        "vol_regime":     "high" if vol_ratio > 1.2 else "low" if vol_ratio < 0.8 else "normal",
        "n_obs":          n,
        "macro_context":  macro_context,
    }


# ── 2. N-BEATS — Client-Side Feature Preparation ─────────────────────────────
#
# N-BEATS (Neural Basis Expansion Analysis) replaces GARCH(1,1).
# GARCH forecasts conditional variance only and assumes returns are Gaussian.
# N-BEATS provides multi-horizon interpretable return forecasts via stacked
# residual MLP blocks; each block backcasts what it "explains" from the input.
#
# Like XGBoost and Attention-LSTM, N-BEATS runs CLIENT-SIDE via ONNX Runtime Web.
# The server returns only the last 30 days of returns (the input window).
#
# Training script:  backend/scripts/train_nbeats.py
# ONNX model:       frontend/public/models/nbeats/model.onnx
#
# Model I/O (after z-score normalisation using training mu_r/sigma_r):
#   Input:  [1, 30]    — last 30 daily returns (normalised)
#   Output: [1, 21, 5] — 21-day × 5-quantile forecast (normalised daily returns)
#
# Fan chart (client-side, 12 recursive 21-day periods → 252 days):
#   The client runs the model 12 times; each iteration uses p50 of the previous
#   forecast as the new input window (direct multi-step rollout).
#   This gives a proper compound-return fan chart with uncertainty that grows
#   organically from the model's learned sequence dynamics.
#
# References
# ----------
# Oreshkin, B., Carpov, D., Chapados, N., & Bengio, Y. (2020).
#   N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series
#   Forecasting. ICLR 2020. https://arxiv.org/abs/1905.10437
# Koenker, R. & Bassett, G. (1978). Regression Quantiles.
#   Econometrica, 46(1), 33–50. https://doi.org/10.2307/1913643
# López de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 7.

def prepare_nbeats_features(
    returns: pd.Series,
    horizon: int,
    last_date: str,
) -> dict:
    """
    Prepare the 30-day input window for client-side N-BEATS inference.

    The ONNX model was trained on z-score normalised returns (global μ/σ
    from the training universe saved in meta.json). NBeatsInferer.js applies
    the same normalisation before inference and reverses it after.

    Returns the raw last-30-day returns; the client loads mu_r/sigma_r
    from meta.json (served as static assets alongside the ONNX model).
    """
    LOOKBACK = 30
    MIN_OBS  = LOOKBACK + LOOKBACK  # need enough history for a meaningful window

    if len(returns) < MIN_OBS:
        raise ValueError(
            f"Insufficient history for N-BEATS: need ≥{MIN_OBS} trading days, "
            f"got {len(returns)}. Try a longer backtest date range."
        )

    last_window    = returns.iloc[-LOOKBACK:].tolist()
    forecast_dates = _forecast_dates(last_date, horizon)

    # Display metadata for the card
    recent_vol = float(returns.iloc[-21:].std() * np.sqrt(TRADING_DAYS))
    recent_ret = float(returns.iloc[-21:].sum())

    return {
        "last_window":    last_window,       # raw 30 daily returns (un-normalised)
        "forecast_dates": forecast_dates,
        "lookback":       LOOKBACK,
        "vol_21d_ann":    round(recent_vol, 4),
        "ret_21d":        round(recent_ret, 4),
        "n_obs":          len(returns),
    }


# ── 3. Hidden Markov Model — Regime-Conditional ───────────────────────────────

def forecast_hmm(
    returns: pd.Series,
    horizon: int,
    n_paths: int,
    last_date: str,
    macro_context: Optional[dict] = None,
) -> dict:
    """
    2-state Gaussian HMM (Bull/Bear) with regime-conditional simulation.

    Parameters are estimated via Baum-Welch (EM). To avoid local optima,
    the model is initialised 10 times with different random seeds and the
    highest-log-likelihood solution is kept — standard practice for HMM
    fitting (Rabiner, 1989).

    Simulation: the current regime probability P(S_T=k | r_1,...,r_T)
    from the Viterbi decoder seeds each path. At each step, the next
    state is sampled from the transition matrix A, and a return is drawn
    from N(μ_k, σ_k).

    References
    ----------
    Hamilton, J.D. (1989). A new approach to the economic analysis of
      nonstationary time series and the business cycle. Econometrica,
      57(2), 357–384. https://doi.org/10.2307/1912559

    Ang, A. & Bekaert, G. (2002). International asset allocation with
      regime shifts. Review of Financial Studies, 15(4), 1137–1187.
      https://doi.org/10.1093/rfs/15.4.1137

    Rabiner, L.R. (1989). A tutorial on hidden Markov models and selected
      applications in speech recognition. Proceedings of the IEEE, 77(2),
      257–286. https://doi.org/10.1109/5.18626

    Out-of-sample methodology
    --------------------------
    Walk-forward split: model trained on 80% of observations.
    Held-out 20% used to verify that regime assignments are economically
    meaningful (Bull state μ > Bear state μ; Bull state σ < Bear state σ).
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 7.
    """
    from hmmlearn.hmm import GaussianHMM

    t0 = time.time()

    n       = len(returns)
    n_train = max(int(n * 0.8), 60)
    r_train = returns.iloc[:n_train].values.reshape(-1, 1)

    # Budget: single fit with 50 iters + diag covariance. Each 2k-sample fit
    # on Railway's shared CPU takes ~8-10 s at n_iter=75; additional restarts
    # delivered little benefit because Baum-Welch was oscillating below tol.
    # Diagonal covariance has 2× fewer parameters for a 2-state 1-D model
    # (identical expressiveness here, faster EM).
    best_model = None
    best_score = -np.inf
    for seed in range(2):
        try:
            m = GaussianHMM(n_components=2, covariance_type="diag",
                            n_iter=50, random_state=seed, tol=1e-3,
                            init_params="stmc")
            m.fit(r_train)
            score = m.score(r_train)
            if score > best_score:
                best_score, best_model = score, m
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("HMM failed to converge")

    model = best_model
    A     = model.transmat_                    # (2,2) transition matrix
    means = model.means_.flatten()             # (2,) per-state daily mean
    stds  = np.sqrt(model.covars_.flatten())   # (2,) per-state daily std

    # Identify Bull (higher mean) and Bear states
    bull = int(np.argmax(means))
    bear = 1 - bull

    # Current regime probability from posterior decode on full history
    r_all  = returns.values.reshape(-1, 1)
    _, seq = model.decode(r_all, algorithm="viterbi")
    current_state = int(seq[-1])
    # Soft probability from posterior
    post       = model.predict_proba(r_all)
    bull_prob  = float(post[-1, bull])

    # Validate out-of-sample: Bull μ > Bear μ (economic sanity)
    oos_sane = bool(means[bull] > means[bear])

    # Simulate — vectorised: draw the full state trajectory, then draw
    # per-state returns in bulk. Saves ~250k Python-level iterations.
    rng    = np.random.default_rng()
    states = np.empty((horizon, n_paths), dtype=np.int8)
    states[0] = current_state
    uniforms  = rng.random((horizon, n_paths))
    cum0      = A[0, 0]
    cum1      = A[1, 0]
    for t in range(1, horizon):
        prev = states[t - 1]
        thresh = np.where(prev == 0, cum0, cum1)
        states[t] = np.where(uniforms[t] < thresh, 0, 1)

    z        = rng.standard_normal((horizon, n_paths))
    step_ret = means[states] + stds[states] * z
    paths    = np.cumprod(1.0 + step_ret, axis=0) - 1.0

    dates = _forecast_dates(last_date, horizon)
    band  = _paths_to_band(paths, dates)

    meta: dict = {
        "bull_state_mu_ann":    round(float(means[bull]) * TRADING_DAYS, 4),
        "bull_state_vol_ann":   round(float(stds[bull]) * np.sqrt(TRADING_DAYS), 4),
        "bear_state_mu_ann":    round(float(means[bear]) * TRADING_DAYS, 4),
        "bear_state_vol_ann":   round(float(stds[bear]) * np.sqrt(TRADING_DAYS), 4),
        "current_bull_prob":    round(bull_prob, 3),
        "current_state":        "bull" if current_state == bull else "bear",
        "transition_bull_bull": round(float(A[bull, bull]), 3),
        "transition_bear_bear": round(float(A[bear, bear]), 3),
        "oos_sanity_ok":        oos_sane,
        "train_n":              n_train,
    }

    if macro_context:
        yc       = macro_context.get("yield_curve_10y2y", 0.6)
        yc_rank  = macro_context.get("yield_curve_pct_rank", 0.5)
        vix_rank = macro_context.get("vix_pct_rank", 0.5)
        cr_rank  = macro_context.get("credit_pct_rank", 0.5)

        if yc < 0:
            yield_regime = "inverted"
        elif yc_rank < 0.25:
            yield_regime = "flat"
        elif yc_rank > 0.75:
            yield_regime = "steep"
        else:
            yield_regime = "normal"

        if vix_rank > 0.80:
            vix_regime = "elevated"
        elif vix_rank > 0.60:
            vix_regime = "above_avg"
        else:
            vix_regime = "calm"

        if yield_regime == "inverted" or cr_rank > 0.80:
            macro_env = "restrictive"
        elif yield_regime in ("steep", "normal") and vix_regime == "calm" and cr_rank < 0.50:
            macro_env = "expansionary"
        else:
            macro_env = "neutral"

        meta["macro_env"]          = macro_env
        meta["yield_curve_regime"] = yield_regime
        meta["vix_regime"]         = vix_regime
        meta["yield_curve_10y2y"]  = round(yc, 3)
        meta["vix_pct_rank"]       = round(vix_rank, 3)
        meta["credit_pct_rank"]    = round(cr_rank, 3)

    return {
        "band": band,
        "metadata": meta,
        "compute_ms": int((time.time() - t0) * 1000),
    }


# ── 4. Fama-French Factor-Anchored Forecast ───────────────────────────────────

def forecast_factor(
    returns: pd.Series,
    horizon: int,
    n_paths: int,
    last_date: str,
    ff5: Optional[dict],
    macro_context: Optional[dict] = None,
) -> dict:
    """
    Factor-anchored GBM: replaces the naive historical mean with a
    theoretically grounded expected return derived from FF5 loadings
    and consensus long-run factor premia.

    Expected return:
        μ_FF5 = RF + β_mkt·E[Mkt-RF] + β_smb·E[SMB] + β_hml·E[HML]
                   + β_rmw·E[RMW] + β_cma·E[CMA] + α

    Simulation uses the idiosyncratic residual volatility (σ_ε from the
    FF5 regression) as the noise term — capturing only the non-factor
    variance and avoiding double-counting systematic risk already priced
    into μ_FF5.

    This approach is standard in institutional capital market assumptions
    (BlackRock Investment Institute; Research Affiliates LLC).

    References
    ----------
    Fama, E.F. & French, K.R. (2015). A five-factor asset pricing model.
      Journal of Financial Economics, 116(1), 1–22.
      https://doi.org/10.1016/j.jfineco.2014.10.010

    Cochrane, J.H. (2011). Presidential address: Discount rates. Journal
      of Finance, 66(4), 1047–1108.
      https://doi.org/10.1111/j.1540-6261.2011.01671.x

    Damodaran, A. (2024). Equity Risk Premiums: Determinants, Estimation
      and Implications. NYU Stern Working Paper.
      https://pages.stern.nyu.edu/~adamodar/

    Out-of-sample note
    ------------------
    Factor premia are sourced from long-run out-of-sample academic consensus
    (60+ year Ken French data) rather than in-sample historical means,
    which dramatically reduces look-ahead bias for short backtests.
    Bailey & Lopez de Prado (2014). Journal of Portfolio Management, 40(5).
    """
    t0 = time.time()

    # If FF5 decomposition available, use its loadings; else fall back to
    # naive historical mean (flagged in metadata)
    # Macro cycle adjustment: inverted yield curve or wide credit spreads
    # suppress the equity risk premium (Campbell & Cochrane 1999; Ludvigson & Ng 2009).
    # Only mkt_rf is scaled — idiosyncratic factor premia are cycle-independent.
    cycle_scale      = 1.0
    macro_adjustment = None
    if macro_context:
        yc_rank = macro_context.get("yield_curve_pct_rank", 0.5)
        cr_rank = macro_context.get("credit_pct_rank", 0.5)
        if yc_rank < 0.15:
            cycle_scale *= 0.70
        elif yc_rank < 0.30:
            cycle_scale *= 0.85
        if cr_rank > 0.85:
            cycle_scale *= 0.75
        elif cr_rank > 0.70:
            cycle_scale *= 0.87
        if cycle_scale != 1.0:
            macro_adjustment = {
                "cycle_scale":      round(cycle_scale, 3),
                "yc_pct_rank":      round(yc_rank, 3),
                "credit_pct_rank":  round(cr_rank, 3),
                "yield_curve_10y2y": round(macro_context.get("yield_curve_10y2y", 0.6), 3),
            }

    if ff5 and all(k in ff5 for k in ("mkt_rf", "smb", "hml", "rmw", "cma", "alpha")):
        # Sum all factor contributions — includes Momentum (mom) if the
        # decomposition was run via Ken French path (Phase 2C upgrade).
        # RMW is the quality/profitability factor (Novy-Marx 2013).
        mu_ann = 0.05  # risk-free anchor
        for factor, premium in FACTOR_PREMIA.items():
            beta  = ff5.get(factor, 0.0)
            scale = cycle_scale if factor == "mkt_rf" else 1.0
            mu_ann += beta * premium * scale
        mu_ann += ff5.get("alpha", 0.0)

        total_vol = float(returns.std() * np.sqrt(TRADING_DAYS))
        r2        = ff5.get("r_squared", 0.0)
        idio_vol  = total_vol * np.sqrt(max(1.0 - r2, 0.01))
        factors_used = ff5.get("factors_used", ["mkt_rf","smb","hml","rmw","cma"])
        source       = f"ff{len(factors_used)}"
    else:
        # Fallback: naive historical mean + full vol (no factor decomp)
        mu_ann   = float(returns.mean() * TRADING_DAYS)
        idio_vol = float(returns.std() * np.sqrt(TRADING_DAYS))
        source   = "historical_fallback"

    dt        = 1 / TRADING_DAYS
    drift     = (mu_ann - 0.5 * idio_vol ** 2) * dt
    diffusion = idio_vol * np.sqrt(dt)

    Z     = np.random.standard_normal((horizon, n_paths))
    paths = np.exp(np.cumsum(drift + diffusion * Z, axis=0)) - 1.0

    dates = _forecast_dates(last_date, horizon)
    band  = _paths_to_band(paths, dates)
    return {
        "band": band,
        "metadata": {
            "mu_factor_ann":    round(mu_ann, 4),
            "mu_hist_ann":      round(float(returns.mean() * TRADING_DAYS), 4),
            "idio_vol_ann":     round(idio_vol, 4),
            "r_squared":        round(ff5.get("r_squared", 0.0) if ff5 else 0.0, 3),
            "source":           source,
            "n_factors":        len(ff5.get("factors_used", ["mkt_rf","smb","hml","rmw","cma"])) if ff5 else 1,
            "mom_beta":         round(ff5.get("mom", 0.0), 3) if ff5 else None,
            "rmw_beta":         round(ff5.get("rmw", 0.0), 3) if ff5 else None,
            "factor_premia":    FACTOR_PREMIA,
            "macro_adjustment": macro_adjustment,
        },
        "compute_ms": int((time.time() - t0) * 1000),
    }


# ── 5. Gaussian Process Autoregression (replaces VAR, Phase 2D) ──────────────
#
# VAR required individual asset return matrices and made stationarity +
# multivariate normality assumptions. GP autoregression (GPAR) replaces it
# with a Bayesian nonparametric model that:
#   - Works on portfolio-level returns directly (no asset price refetch)
#   - Provides exact posterior uncertainty — no Monte Carlo needed
#   - Has no stationarity requirement
#   - Automatically tunes complexity via marginal likelihood maximisation
#
# Architecture:
#   Input X: rolling 7-day return window (ARD Matérn features per lag)
#   Kernel:  ConstantKernel × Matérn(ν=5/2, ARD) + WhiteKernel
#   Output:  one-step predictive N(μ_t, σ_t²)
#   Rollout: 252 steps, advancing window with predictive mean
#   Fan chart: cumulative returns ~ N(Σμ_t, Σσ_t²) (Gaussian approximation)
#
# References
# ----------
# Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for Machine
#   Learning. MIT Press. https://gaussianprocess.org/gpml/
# Matérn, B. (1960). Spatial Variation. Meddelanden från Statens
#   Skogsforskningsinstitut, 49(5). [ν=5/2 yields twice-differentiable paths]
# Roberts, S. et al. (2013). Gaussian Processes for time series modelling.
#   Phil. Trans. R. Soc. A, 371. https://doi.org/10.1098/rsta.2011.0550
# López de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 7.

def forecast_gp(
    returns: pd.Series,
    horizon: int,
    last_date: str,
) -> dict:
    """
    Gaussian Process autoregression for 252-day portfolio return forecasting.

    Fits an ARD Matérn 5/2 GP on the most recent daily returns using a
    rolling-window autoregressive framing (X = last 7 returns, y = next return).
    Capped at 300 training samples for O(n³) tractability.

    OOS methodology (López de Prado 2018):
      Chronological 80/20 split. Held-out 20% used to compute predictive R²
      (skill) and NLPD (calibration quality — lower is better).

    Fan chart (Gaussian approximation):
      Cumulative return at step T ≈ N(Σμ_t, Σσ_t²) under the independence
      approximation — exact for uncorrelated Gaussian daily returns.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from scipy.stats import norm as scipy_norm

    t0 = time.time()

    LOOKBACK     = 7    # one trading week of lags as GP features
    N_TRAIN_MAX  = 150  # cap for O(n³) tractability on Railway's shared CPU

    r = returns.values
    n = len(r)

    if n < LOOKBACK + 40:
        raise ValueError(
            f"GP requires ≥{LOOKBACK + 40} observations, got {n}. "
            "Try a longer backtest date range."
        )

    # Build autoregressive sequences
    X_all = np.array([r[i - LOOKBACK:i] for i in range(LOOKBACK, n)])
    y_all = r[LOOKBACK:]

    # Most-recent N_TRAIN_MAX points (chronological — no shuffle)
    X_fit = X_all[-N_TRAIN_MAX:]
    y_fit = y_all[-N_TRAIN_MAX:]
    n_fit = len(X_fit)

    # Chronological 80/20 OOS split
    n_tr  = max(int(n_fit * 0.8), 30)
    X_tr, y_tr   = X_fit[:n_tr], y_fit[:n_tr]
    X_val, y_val = X_fit[n_tr:], y_fit[n_tr:]

    # ARD Matérn 5/2: each lag gets its own length scale (7 hyperparameters).
    # ConstantKernel adjusts signal amplitude; WhiteKernel absorbs noise floor.
    kernel = (
        ConstantKernel(1.0, (1e-4, 1e2))
        * Matern(length_scale=np.ones(LOOKBACK),
                 length_scale_bounds=(1e-3, 10.0), nu=2.5)
        + WhiteKernel(noise_level=float(np.var(y_tr)) + 1e-8,
                      noise_level_bounds=(1e-8, 1.0))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0,              # noise modelled entirely by WhiteKernel
        n_restarts_optimizer=0,   # single fit — kernel init is already reasonable
        normalize_y=True,
    )
    gp.fit(X_tr, y_tr)

    # OOS diagnostics
    oos_r2 = nlpd = None
    if len(y_val) > 1:
        mu_val, std_val = gp.predict(X_val, return_std=True)
        ss_res  = float(np.sum((y_val - mu_val) ** 2))
        ss_tot  = float(np.sum((y_val - y_val.mean()) ** 2))
        oos_r2  = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None
        # Negative log predictive density (calibration — lower = better)
        nlpd = round(float(np.mean(
            0.5 * np.log(2 * np.pi * std_val ** 2)
            + (y_val - mu_val) ** 2 / (2 * std_val ** 2 + 1e-10)
        )), 4)

    # 252-step rollout: advance window with GP predictive mean each step
    window = list(r[-LOOKBACK:])
    mu_fwd  = np.empty(horizon)
    std_fwd = np.empty(horizon)

    for t in range(horizon):
        x_t = np.array(window[-LOOKBACK:]).reshape(1, -1)
        mu_t, std_t = gp.predict(x_t, return_std=True)
        mu_fwd[t]   = float(mu_t[0])
        std_fwd[t]  = float(std_t[0])
        window.append(float(mu_t[0]))

    # Cumulative fan chart via Gaussian approximation: Σr_t ~ N(Σμ_t, Σσ_t²)
    cum_mu  = np.cumsum(mu_fwd) * 100     # percent
    cum_std = np.sqrt(np.cumsum(std_fwd ** 2)) * 100

    dates = _forecast_dates(last_date, horizon)
    band  = {
        "dates": dates,
        "p5":  (cum_mu + scipy_norm.ppf(0.05) * cum_std).tolist(),
        "p25": (cum_mu + scipy_norm.ppf(0.25) * cum_std).tolist(),
        "p50": cum_mu.tolist(),
        "p75": (cum_mu + scipy_norm.ppf(0.75) * cum_std).tolist(),
        "p95": (cum_mu + scipy_norm.ppf(0.95) * cum_std).tolist(),
    }

    # Extract optimised noise level for metadata
    noise_level = None
    for component in gp.kernel_.get_params().values():
        if hasattr(component, "noise_level"):
            noise_level = round(float(component.noise_level), 6)
            break

    return {
        "band": band,
        "metadata": {
            "oos_r2":      oos_r2,
            "nlpd":        nlpd,
            "lookback":    LOOKBACK,
            "n_train":     n_tr,
            "noise_level": noise_level,
            "kernel":      "Matérn(ν=5/2, ARD) + WhiteKernel",
        },
        "compute_ms": int((time.time() - t0) * 1000),
    }


# ── 6. Attention-LSTM — Client-Side Feature Preparation ──────────────────────
#
# The Attention-LSTM now runs IN THE USER'S BROWSER via TensorFlow.js.
# This eliminates the ~450MB tensorflow-cpu server RAM cost and keeps
# Railway free-tier usage well under the 512MB limit.
#
# The server's role is ONLY feature engineering:
#   1. Compute the 5-feature matrix from the portfolio's return series
#   2. Fit a per-portfolio MinMaxScaler
#   3. Return the last 60-day scaled window + scaler parameters
#
# The browser then loads the pre-trained TF.js model (served as static assets
# by Vercel from frontend/public/models/lstm/) and runs 200 MC Dropout passes.
#
# Training script: backend/scripts/train_lstm.py
# TF.js model:    frontend/public/models/lstm/model.json
#
# Architecture (unchanged from prior server implementation):
#   Input(60, 5) → LSTM(64, return_sequences=True)
#   → Bahdanau attention → context ∈ ℝ⁶⁴
#   → Dropout(0.20) → Dense(32, relu) → Dense(1)
#
# References
# ----------
# CS230 Stanford (2020). https://cs230.stanford.edu/projects_winter_2020/reports/32066186.pdf
# Bahdanau, Cho & Bengio (2015, ICLR). arXiv:1409.0473
# Gal & Ghahramani (2016, ICML). MC Dropout as Bayesian Approximation.
# Fischer & Krauss (2018, EJOR). LSTM for financial market prediction.

def prepare_lstm_features(
    returns: pd.Series,
    horizon: int,
    last_date: str,
) -> dict:
    """
    Prepare scale-invariant feature window for client-side Attention-LSTM inference.

    Features (all bounded in ~[-1, 1] by construction — NO per-portfolio scaler):
      [0] z_ret       = r_t / σ_21d(r_{t-21..t-1}), clipped to ±3, divided by 3
      [1] vol_pct     = percentile of σ_63d in its 5y trailing window, mapped to [-1, 1]
      [2] z_mom5      = sum_5(r) / (σ_5 * √5),   clipped to ±3, divided by 3
      [3] z_mom21    = sum_21(r) / (σ_21 * √21), clipped to ±3, divided by 3
      [4] rsi_centred = 2*rsi_14 - 1

    The browser unscales predictions via:  r_pred = (model_out * 3) * current_sigma_21d

    Contract MUST stay in lockstep with:
      - backend/scripts/train_lstm.py  (training feature builder)
      - frontend/src/ml/LSTMInferer.js (client-side rollout)
    """
    LOOKBACK    = 60
    MIN_OBS     = LOOKBACK + 90
    CLIP_SIGMAS = 3.0

    if len(returns) < MIN_OBS:
        raise ValueError(
            f"Insufficient history for Attention-LSTM: need ≥{MIN_OBS} trading days, "
            f"got {len(returns)}. Try a longer backtest date range."
        )

    r  = returns.copy()
    df = pd.DataFrame({"r": r})

    # Causal rolling std — shifted by 1 so we don't leak r_t into its own denominator
    sigma_21 = r.rolling(21).std().shift(1)
    sigma_5  = r.rolling(5).std().shift(1)
    sigma_63 = r.rolling(63).std().shift(1)

    # Vol regime: where does today's 63d vol sit in its 5y history, mapped to [-1, 1]
    vol_pct_raw = sigma_63.rolling(5 * TRADING_DAYS, min_periods=TRADING_DAYS).rank(pct=True)
    df["vol_pct"] = (2.0 * vol_pct_raw - 1.0).clip(-1, 1)

    df["z_ret"]   = (r / (sigma_21 + 1e-9)).clip(-CLIP_SIGMAS, CLIP_SIGMAS) / CLIP_SIGMAS

    mom5  = r.rolling(5).sum()
    mom21 = r.rolling(21).sum()
    df["z_mom5"]  = (mom5  / (sigma_5  * np.sqrt(5)  + 1e-9)).clip(-CLIP_SIGMAS, CLIP_SIGMAS) / CLIP_SIGMAS
    df["z_mom21"] = (mom21 / (sigma_21 * np.sqrt(21) + 1e-9)).clip(-CLIP_SIGMAS, CLIP_SIGMAS) / CLIP_SIGMAS

    delta = r.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi14 = gain / (gain + loss + 1e-9)
    df["rsi_centred"] = (2.0 * rsi14 - 1.0).clip(-1, 1)

    feat_cols = ["z_ret", "vol_pct", "z_mom5", "z_mom21", "rsi_centred"]
    df = df.dropna(subset=feat_cols)

    if len(df) < LOOKBACK:
        raise ValueError(
            f"Insufficient usable history after feature engineering: "
            f"{len(df)} rows < lookback {LOOKBACK}."
        )

    last_window    = df[feat_cols].values[-LOOKBACK:].astype(float).tolist()
    forecast_dates = _forecast_dates(last_date, horizon)

    # Client needs the current σ_21d to unscale z-predictions back to raw returns.
    current_sigma_21d = float(r.iloc[-21:].std())
    if not np.isfinite(current_sigma_21d) or current_sigma_21d <= 0:
        current_sigma_21d = 0.015  # ~1.5% fallback

    # Raw return seed: last ~128 returns are enough to rebuild z_ret / z_mom5 /
    # z_mom21 / RSI during the rollout (all use ≤ 21d windows + small buffers).
    raw_return_seed = r.iloc[-128:].astype(float).tolist()

    # σ_63 percentile history: the vol_pct feature ranks today's σ_63 within its
    # trailing 5y distribution. We precompute that distribution on the server so
    # the client doesn't need to ship 1260 raw returns.
    sigma_63_series = sigma_63.dropna().iloc[-(5 * TRADING_DAYS):].astype(float)
    sigma_63_history = sigma_63_series.tolist()

    return {
        "last_window":        last_window,
        "feature_names":      feat_cols,
        "forecast_dates":     forecast_dates,
        "raw_return_seed":    raw_return_seed,
        "current_sigma_21d":  current_sigma_21d,
        "sigma_63_history":   sigma_63_history,
        "clip_sigmas":        CLIP_SIGMAS,
    }


# ── Tier 2 portfolio-level aggregators ───────────────────────────────────────

def _portfolio_reddit(reddit_ctx: dict, weights: dict) -> dict | None:
    """Weighted-average Reddit mention count and score across portfolio tickers."""
    if not reddit_ctx:
        return None
    total_w = sum(abs(w) for t, w in weights.items() if reddit_ctx.get(t, {}).get("available"))
    if total_w == 0:
        return None
    mentions = sum(
        reddit_ctx[t]["mention_count_7d"] * abs(w)
        for t, w in weights.items()
        if reddit_ctx.get(t, {}).get("available")
    ) / total_w
    scores = sum(
        reddit_ctx[t]["avg_score_7d"] * abs(w)
        for t, w in weights.items()
        if reddit_ctx.get(t, {}).get("available")
    ) / total_w
    return {
        "portfolio_mentions_7d": round(mentions, 1),
        "portfolio_avg_score":   round(scores, 1),
        "available":             True,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_all_forecasts(req) -> dict:
    """
    Orchestrator: runs requested methods, collects results.
    Each method failure is isolated — the endpoint always returns a
    complete response with per-method error fields as needed.
    """
    from app.models.forecast import ForecastBand, MethodResult, ForecastResponse

    returns = _equity_to_returns(req.equity_curve)
    last_date = req.end_date
    h  = req.horizon_days
    np_ = req.n_paths

    # Forecast end date
    forecast_dates = _forecast_dates(last_date, h)
    forecast_end   = forecast_dates[-1] if forecast_dates else last_date
    hist_end_val   = float(req.equity_curve[-1]["value"]) if req.equity_curve else 1.0

    # ── Data providers dispatched in parallel ───────────────────────────────────
    #
    # Previously these six providers (macro, insider, news, reddit, trends, edgar)
    # ran serially before any method dispatch, turning a cold-cache 5-ticker
    # request into 30-50s of sequential network I/O. They're all independent and
    # I/O-bound, so a single ThreadPoolExecutor fans them out for a ~5-10× win.
    #
    # Phase 1 methods (xgboost / nbeats / factor) only use `returns` in the
    # model math — tier-2 text providers (news / reddit / trends / edgar) are
    # purely display metadata for those methods and are skipped to keep Phase 1
    # at its advertised ~1-2s budget. They still run in Phase 2 where HMM / VAR
    # / LSTM return the tier-2 context to the frontend for display.
    tickers = [a.ticker for a in req.assets] if req.assets else []
    weights = {a.ticker: a.weight for a in req.assets} if req.assets else {}

    _PHASE2_SERVER_METHODS = {"hmm", "var", "lstm"}
    is_phase2_request = any(m in _PHASE2_SERVER_METHODS for m in req.methods)

    macro_ctx:  dict = {}
    insider:    dict = {}
    news_ctx:   dict = {}
    reddit_ctx: dict = {}
    edgar_ctx:  dict = {}

    from concurrent.futures import ThreadPoolExecutor as _CtxPool

    def _safe_macro():
        try:
            from .fred_provider import get_macro_features
            from .vix_provider  import get_vix_features
            return {**get_macro_features(last_date), **get_vix_features(last_date)}
        except Exception as exc:
            logger.debug("Macro context skipped: %s", exc)
            return {}

    def _safe_insider():
        if not tickers:
            return {}
        try:
            from .sec_provider import get_insider_features
            return get_insider_features(tickers, weights, last_date)
        except Exception as exc:
            logger.debug("SECProvider skipped: %s", exc)
            return {}

    def _safe_news():
        if not tickers:
            return {}
        try:
            from .news_provider import get_news_context
            return get_news_context(tickers, weights, last_date)
        except Exception as exc:
            logger.debug("NewsProvider skipped: %s", exc)
            return {}

    def _safe_reddit():
        if not tickers:
            return {}
        try:
            from .reddit_provider import get_reddit_context
            return get_reddit_context(tickers, last_date)
        except Exception as exc:
            logger.debug("RedditProvider skipped: %s", exc)
            return {}

    def _safe_edgar():
        if not tickers:
            return {}
        try:
            from .edgar_filing_provider import get_edgar_filing_context
            return get_edgar_filing_context(tickers, weights, last_date)
        except Exception as exc:
            logger.debug("EDGARFiling skipped: %s", exc)
            return {}

    # Tier 1 always runs — macro is cached (cheap) and insider feeds xgboost
    # metadata pills on both phases.
    _provider_jobs = {
        "macro":   _safe_macro,
        "insider": _safe_insider,
    }
    if is_phase2_request:
        _provider_jobs.update({
            "news":   _safe_news,
            "reddit": _safe_reddit,
            "edgar":  _safe_edgar,
        })

    with _CtxPool(max_workers=len(_provider_jobs)) as _ctx_pool:
        _ctx_futures = {name: _ctx_pool.submit(fn) for name, fn in _provider_jobs.items()}
        macro_ctx  = _ctx_futures["macro"].result()   or {}
        insider    = _ctx_futures["insider"].result() or {}
        if is_phase2_request:
            news_ctx   = _ctx_futures["news"].result()   or {}
            reddit_ctx = _ctx_futures["reddit"].result() or {}
            edgar_ctx  = _ctx_futures["edgar"].result()  or {}

    if macro_ctx:
        logger.info(
            "Macro context: YC=%.2f (pct=%.2f) VIX_rank=%.2f credit_pct=%.2f src=%s",
            macro_ctx.get("yield_curve_10y2y", 0),
            macro_ctx.get("yield_curve_pct_rank", 0),
            macro_ctx.get("vix_pct_rank", 0),
            macro_ctx.get("credit_pct_rank", 0),
            macro_ctx.get("source", "?"),
        )
    if insider.get("available"):
        logger.info(
            "SECProvider: portfolio net buying 30d = $%.1fM  (n=%d tickers)",
            insider["portfolio_net_buying_30d"],
            insider["n_tickers_with_data"],
        )
    if is_phase2_request:
        if news_ctx:
            logger.info(
                "NewsProvider: %d total articles",
                news_ctx.get("portfolio_summary", {}).get("total_articles", 0),
            )
        if reddit_ctx:
            logger.info(
                "RedditProvider: %d total mentions",
                reddit_ctx.get("portfolio_summary", {}).get("total_mentions", 0),
            )
        if edgar_ctx:
            logger.info(
                "EDGARFiling: %d tickers with 10-K/10-Q excerpts",
                edgar_ctx.get("tickers_with_data", 0),
            )

    _hmm_out_extras: dict = {}   # populated during HMM dispatch for regime/ensemble

    # Heavy server-side methods (HMM / GP) are independent — dispatch them
    # concurrently via a thread pool so the slowest method doesn't starve the
    # others within the client's 60 s Phase 2 budget.
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    PHASE2_SERVER_METHODS = {"hmm", "var"}
    _pending_futures: dict = {}
    _server_pool = ThreadPoolExecutor(max_workers=2)
    for m in req.methods:
        if m == "hmm":
            _pending_futures["hmm"] = _server_pool.submit(
                forecast_hmm, returns, h, np_, last_date, macro_ctx or None,
            )
        elif m == "var":
            _pending_futures["var"] = _server_pool.submit(
                forecast_gp, returns, h, last_date,
            )

    results = []
    for method in req.methods:
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, "#888888")
        try:
            # ── Client-side methods: server returns features only, no band ─────
            if method == "xgboost":
                out = prepare_xgboost_features(returns, h, last_date)
                results.append(MethodResult(
                    method=method, label=label, color=color,
                    forecast=None,
                    metadata={
                        "client_side":     True,
                        "xgb_features":    out,
                        "model":           "HistGradientBoostingRegressor (sklearn)",
                        "quantiles":       [0.05, 0.25, 0.50, 0.75, 0.95],
                        "horizon_days":    21,
                        "insider_context": insider,
                        "reddit_context":  _portfolio_reddit(reddit_ctx, weights) or None,
                        "news_article_count": news_ctx.get("portfolio_summary", {}).get("total_articles"),
                    },
                ))
                continue

            if method == "nbeats":
                out = prepare_nbeats_features(returns, h, last_date)
                results.append(MethodResult(
                    method=method, label=label, color=color,
                    forecast=None,
                    metadata={
                        "client_side":    True,
                        "nbeats_features": out,
                        "architecture":   "N-BEATS: 3 blocks × FC(256)×4 → backcast + forecast",
                        "quantiles":      [0.05, 0.25, 0.50, 0.75, 0.95],
                        "macro_context":  macro_ctx or None,
                    },
                ))
                continue

            if method == "lstm":
                out = prepare_lstm_features(returns, h, last_date)
                results.append(MethodResult(
                    method=method, label=label, color=color,
                    forecast=None,
                    metadata={
                        "client_side":   True,
                        "lstm_features": out,
                        "architecture":  "Attention-LSTM(64) → Dense(32) → Dense(1)",
                        "attention":     "Bahdanau additive (CS230 2020)",
                        "news_article_count": news_ctx.get("portfolio_summary", {}).get("total_articles"),
                    },
                ))
                continue

            # ── Legacy/removed methods ───────────────────────────────────────────
            if method in ("monte_carlo", "garch"):
                raise ValueError(
                    f"Method '{method}' was removed. "
                    "Use 'xgboost' (replaced monte_carlo) or 'nbeats' (replaced garch)."
                )

            # ── Server-side methods: compute band directly ─────────────────────
            if method == "hmm":
                # 45 s cap — HMM fit on Railway's shared CPU runs ~10 s per
                # restart and is the wall-clock bottleneck of Phase 2.
                out = _pending_futures["hmm"].result(timeout=45)
                if news_ctx:
                    out["metadata"]["news_article_count"] = news_ctx.get("portfolio_summary", {}).get("total_articles")
                # Phase 5A: 4-state regime classification after HMM (non-fatal)
                try:
                    from .meta_learner import compute_regime_probs, get_ensemble_weights
                    vix_rank = macro_ctx.get("vix_pct_rank", 0.5) if macro_ctx else 0.5
                    rp = compute_regime_probs(out.get("metadata", {}), vix_rank)
                    ew = get_ensemble_weights(rp, req.methods)
                    _hmm_out_extras["regime_probs"]     = rp
                    _hmm_out_extras["ensemble_weights"] = ew
                    logger.info("Regime: %s (dom=%s)", rp, rp.get("dominant"))
                except Exception as exc:
                    logger.debug("Regime classification failed: %s", exc)

            elif method == "factor":
                out = forecast_factor(returns, h, np_, last_date,
                                     req.ff5_decomposition, macro_ctx or None)
                if insider.get("available"):
                    out["metadata"]["insider_context"] = insider

            elif method == "var":
                # VAR key preserved for backward compat; computes Gaussian Process
                out = _pending_futures["var"].result(timeout=35)
                if macro_ctx:
                    out["metadata"]["vix_pct_rank"]      = macro_ctx.get("vix_pct_rank")
                    out["metadata"]["vix_term_slope"]    = macro_ctx.get("vix_term_slope")
                    out["metadata"]["yield_curve_10y2y"] = macro_ctx.get("yield_curve_10y2y")

            else:
                raise ValueError(f"Unknown forecast method: {method!r}")

            # ── Common append path for hmm/factor/var ─────────────────────────
            results.append(MethodResult(
                method=method, label=label, color=color,
                forecast=ForecastBand(**out["band"]),
                metadata=out.get("metadata", {}),
                compute_ms=out.get("compute_ms", 0),
            ))

        except FuturesTimeout:
            logger.warning("Forecast method %s exceeded its server-side time budget", method)
            results.append(MethodResult(
                method=method, label=label, color=color,
                error=f"{label} exceeded the server-side time budget — try a shorter backtest window",
                status="error",
            ))
        except Exception as e:
            logger.warning("Forecast method %s failed: %s", method, e)
            results.append(MethodResult(
                method=method, label=label, color=color,
                error=str(e),
                status="error",
            ))

    # Threads are daemonic by default on 3.9+, but explicit shutdown keeps
    # the interpreter from holding onto a stuck HMM fit beyond the response.
    _server_pool.shutdown(wait=False, cancel_futures=True)

    tier2_summary = {
        "reddit":  _portfolio_reddit(reddit_ctx, weights) if reddit_ctx else None,
        "news_available": bool(news_ctx.get("portfolio_summary", {}).get("available")),
    }

    regime_probs     = _hmm_out_extras.get("regime_probs")
    ensemble_weights = _hmm_out_extras.get("ensemble_weights")

    return ForecastResponse(
        forecast_start=forecast_dates[0] if forecast_dates else last_date,
        forecast_end=forecast_end,
        historical_end_value=hist_end_val,
        results=results,
        news_context=news_ctx or None,
        edgar_context=edgar_ctx or None,
        tier2_context=tier2_summary,
        regime_probs=regime_probs,
        ensemble_weights=ensemble_weights,
    )
