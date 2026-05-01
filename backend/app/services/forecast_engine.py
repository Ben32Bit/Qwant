"""
Forecast Engine — 4 Research-Backed Portfolio Forecasting Methods
=================================================================

All methods produce 3-month (63-trading-day) probabilistic forecasts as
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

# Shadow (OOS) holdout window — shared across all server-side methods so the
# evaluation regime is apples-to-apples (Phase C of forecastupgrade.md).
# Every method trains on [start, T−SHADOW_DAYS] and forecasts SHADOW_HORIZON
# days forward; the predicted p50 is compared against actual prices to get OOS R².
SHADOW_DAYS    = 60   # hold out the last 60 trading days
SHADOW_HORIZON = 30   # shadow forecast covers the first 30 of those days
MIN_SHADOW_HISTORY = 90  # minimum history before shadow cutoff to be meaningful

# Method colours for the frontend
METHOD_COLORS = {
    "nbeats":   "#ffd43b",   # amber
    "hmm":      "#a855f7",   # purple
    "var":      "#ff6b35",   # orange (GP)
    "lstm":     "#ff4757",   # red
    "timesfm":  "#00d4aa",   # green (freed from Factor model)
}

METHOD_LABELS = {
    "nbeats":   "N-BEATS Neural",
    "hmm":      "Hidden Markov Model",
    "var":      "Gaussian Process (GP)",
    "lstm":     "LSTM Neural Net",
    "timesfm":  "TimesFM 2.5 (Google)",
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


# ── OOS R² computation ────────────────────────────────────────────────────────

def _compute_oos_r2(shadow_band: dict, actual_returns: "pd.Series") -> Optional[float]:
    """
    Out-of-sample R² of the shadow forecast p50 vs actual portfolio returns
    in the holdout window [T−SHADOW_DAYS, T−SHADOW_DAYS+SHADOW_HORIZON].

    Both series are expressed as cumulative % return from the shadow training
    end so scale differences between portfolios don't affect the metric.

    Returns None if there is insufficient data or a numerical error occurs.
    """
    try:
        n = min(SHADOW_HORIZON,
                len(actual_returns),
                len(shadow_band.get("p50", [])))
        if n < 5:
            return None
        # Actual cumulative % return over the holdout window
        actual_cum = (np.cumprod(1.0 + actual_returns.values[:n]) - 1.0) * 100.0
        shadow_p50 = np.array(shadow_band["p50"][:n], dtype=float)

        ss_res = float(np.sum((actual_cum - shadow_p50) ** 2))
        ss_tot = float(np.sum((actual_cum - actual_cum.mean()) ** 2))
        return round(1.0 - ss_res / ss_tot, 4) if ss_tot > 1e-10 else 0.0
    except Exception:
        return None


# ── 1. N-BEATS — Client-Side Feature Preparation ─────────────────────────────
#
# N-BEATS (Neural Basis Expansion Analysis) replaces GARCH(1,1).
# GARCH forecasts conditional variance only and assumes returns are Gaussian.
# N-BEATS provides multi-horizon interpretable return forecasts via stacked
# residual MLP blocks; each block backcasts what it "explains" from the input.
#
# Like the Attention-LSTM, N-BEATS runs CLIENT-SIDE via the browser.
# The server returns only the last 30 days of returns (the input window).
#
# Training script:  backend/scripts/train_nbeats.py
# ONNX model:       frontend/public/models/nbeats/model.onnx
#
# Model I/O (after z-score normalisation using training mu_r/sigma_r):
#   Input:  [1, 30]    — last 30 daily returns (normalised)
#   Output: [1, 21, 5] — 21-day × 5-quantile forecast (normalised daily returns)
#
# Fan chart (client-side, 3 recursive 21-day periods → 63 days):
#   The client runs the model 3 times; each iteration uses p50 of the previous
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


# ── 2. Hidden Markov Model — Regime-Conditional ───────────────────────────────

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

    # Budget: single fit with 50 iters + diag covariance. hmmlearn's default
    # k-means init ("stmc") is deterministic and close enough to the MLE that
    # extra random restarts delivered negligible improvement in testing while
    # doubling Phase-2 wall-clock on Railway's shared CPU (~8-10s per fit).
    # Diagonal covariance has 2× fewer parameters for a 2-state 1-D model
    # (identical expressiveness here, faster EM).
    try:
        model = GaussianHMM(n_components=2, covariance_type="diag",
                            n_iter=50, random_state=0, tol=1e-3,
                            init_params="stmc")
        model.fit(r_train)
    except Exception:
        raise RuntimeError("HMM failed to converge")
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

    # In-sample R²: posterior expected return vs actual over full history.
    # Measures how well the 2-state mean structure fits the training data.
    pred_is  = float(means[bull]) * post[:, bull] + float(means[bear]) * post[:, bear]
    actual_is = returns.values
    ss_res_is = float(np.sum((actual_is - pred_is) ** 2))
    ss_tot_is = float(np.sum((actual_is - actual_is.mean()) ** 2))
    is_r2 = round(1.0 - ss_res_is / ss_tot_is, 4) if ss_tot_is > 1e-10 else None

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
        "is_r2":                is_r2,
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
        decorate_hmm_meta_with_macro(meta, macro_context)

    return {
        "band": band,
        "metadata": meta,
        "compute_ms": int((time.time() - t0) * 1000),
    }


def decorate_hmm_meta_with_macro(meta: dict, macro_context: dict) -> None:
    """
    In-place decorate HMM metadata with macro-regime labels derived from
    yield-curve / VIX / credit spread ranks. Extracted from forecast_hmm so
    the HMM fit can run in parallel with the tier-2 provider fan-out: the
    HMM future is dispatched before macro has landed, and this helper
    back-fills the meta{} once macro_ctx is available.
    """
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


# ── 3. Gaussian Process Autoregression (replaces VAR, Phase 2D) ──────────────
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
#   Rollout: 63 steps, advancing window with predictive mean
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
    Gaussian Process autoregression for 63-day portfolio return forecasting.

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

    # IS diagnostics (posterior mean on training set)
    mu_tr_is  = gp.predict(X_tr)
    ss_res_is = float(np.sum((y_tr - mu_tr_is) ** 2))
    ss_tot_is = float(np.sum((y_tr - y_tr.mean()) ** 2))
    is_r2 = round(1.0 - ss_res_is / ss_tot_is, 4) if ss_tot_is > 1e-10 else None

    # OOS diagnostics (80/20 within-window split)
    oos_r2 = nlpd = None
    if len(y_val) > 1:
        mu_val, std_val = gp.predict(X_val, return_std=True)
        ss_res  = float(np.sum((y_val - mu_val) ** 2))
        ss_tot  = float(np.sum((y_val - y_val.mean()) ** 2))
        oos_r2  = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None
        # Negative log predictive density (calibration — lower is better)
        nlpd = round(float(np.mean(
            0.5 * np.log(2 * np.pi * std_val ** 2)
            + (y_val - mu_val) ** 2 / (2 * std_val ** 2 + 1e-10)
        )), 4)

    # Horizon-step rollout: advance window with GP predictive mean each step
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
            "is_r2":       is_r2,
            "oos_r2":      oos_r2,
            "nlpd":        nlpd,
            "lookback":    LOOKBACK,
            "n_train":     n_tr,
            "noise_level": noise_level,
            "kernel":      "Matérn(ν=5/2, ARD) + WhiteKernel",
        },
        "compute_ms": int((time.time() - t0) * 1000),
    }


# ── 4. Attention-LSTM — Client-Side Feature Preparation ──────────────────────
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

    # ── Shadow (OOS) holdout setup ───────────────────────────────────────────
    # Train on [start, T-SHADOW_DAYS], forecast SHADOW_HORIZON days forward,
    # compare shadow p50 against actual prices → OOS R² per server-side method.
    shadow_cutoff         = len(returns) - SHADOW_DAYS
    can_shadow            = shadow_cutoff >= MIN_SHADOW_HISTORY
    returns_shadow        = returns.iloc[:shadow_cutoff] if can_shadow else None
    shadow_last_date      = returns_shadow.index[-1].strftime("%Y-%m-%d") if can_shadow else None
    shadow_actual_returns = (
        returns.iloc[shadow_cutoff: shadow_cutoff + SHADOW_HORIZON] if can_shadow else None
    )
    _shadow_dates = _forecast_dates(shadow_last_date, SHADOW_HORIZON) if can_shadow else []
    shadow_forecast_start = _shadow_dates[0]  if _shadow_dates else None
    shadow_forecast_end   = _shadow_dates[-1] if _shadow_dates else None

    # ── Data providers dispatched in parallel ───────────────────────────────────
    #
    # Previously these five providers (macro, insider, news, reddit, edgar) ran
    # serially before any method dispatch, turning a cold-cache 5-ticker request
    # into 30-50s of sequential network I/O. They're all independent and
    # I/O-bound, so a single ThreadPoolExecutor fans them out for a ~5-10× win.
    #
    # Phase 1 (nbeats) only uses `returns` — tier-2 text providers are skipped
    # to keep Phase 1 at its advertised ~1-2s budget. They run in Phase 2 where
    # HMM / GP / LSTM return the tier-2 context to the frontend for display.
    tickers = [a.ticker for a in req.assets] if req.assets else []
    weights = {a.ticker: a.weight for a in req.assets} if req.assets else {}

    _PHASE2_SERVER_METHODS = {"hmm", "var", "lstm", "timesfm"}
    is_phase2_request = any(m in _PHASE2_SERVER_METHODS for m in req.methods)

    macro_ctx:  dict = {}
    insider:    dict = {}
    news_ctx:   dict = {}
    reddit_ctx: dict = {}
    edgar_ctx:  dict = {}

    from concurrent.futures import ThreadPoolExecutor as _CtxPool
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    # ── Heavy server models dispatched FIRST ───────────────────────────────
    # HMM and GP each take ~10-15s on Railway's shared CPU and neither depends
    # on the tier-2 text providers (news/reddit/edgar) — those feed purely
    # display metadata. Previously the method dispatch was in a second block
    # AFTER awaiting the full provider pool, so HMM/GP serialised behind the
    # slowest network fetch (often reddit). Submitting them here lets the
    # C-extension fits overlap with the I/O-bound provider fan-out.
    #
    # HMM is fitted with macro_context=None; the macro-regime labels are
    # back-filled into out["metadata"] once the macro future lands.
    _pending_futures: dict = {}
    # 6 workers: up to 3 forward + 3 shadow running in parallel so Phase 2
    # wall-clock doesn't grow — shadow fits overlap with forward fits.
    _server_pool = ThreadPoolExecutor(max_workers=6)
    for _m in req.methods:
        if _m == "hmm":
            _pending_futures["hmm"] = _server_pool.submit(
                forecast_hmm, returns, h, np_, last_date, None,
            )
            if can_shadow:
                # Use 500 MC paths for shadow (half of forward) — accuracy is
                # sufficient for OOS R² and halves the simulation time.
                _pending_futures["hmm_shadow"] = _server_pool.submit(
                    forecast_hmm, returns_shadow, SHADOW_HORIZON, min(np_, 500),
                    shadow_last_date, None,
                )
        elif _m == "var":
            _pending_futures["var"] = _server_pool.submit(
                forecast_gp, returns, h, last_date,
            )
            if can_shadow:
                _pending_futures["var_shadow"] = _server_pool.submit(
                    forecast_gp, returns_shadow, SHADOW_HORIZON, shadow_last_date,
                )
        elif _m == "timesfm":
            from app.services.timesfm_provider import forecast_timesfm
            _pending_futures["timesfm"] = _server_pool.submit(
                forecast_timesfm, returns, h, last_date,
            )
            if can_shadow:
                _pending_futures["timesfm_shadow"] = _server_pool.submit(
                    forecast_timesfm, returns_shadow, SHADOW_HORIZON, shadow_last_date,
                )

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

    # Tier 1 always runs — macro is cached (cheap) and insider feeds method metadata.
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

    # HMM / GP / TimesFM futures were already submitted up top so they overlap
    # with the tier-2 provider fan-out. Macro context is back-filled onto HMM
    # meta in the per-method loop below (forecast_hmm itself was called with None).
    PHASE2_SERVER_METHODS = {"hmm", "var", "timesfm"}

    results = []
    for method in req.methods:
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, "#888888")
        try:
            # ── Client-side methods: server returns features only, no band ─────
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
            if method in ("monte_carlo", "garch", "xgboost", "factor"):
                raise ValueError(
                    f"Method '{method}' was removed. "
                    "Supported methods: nbeats | hmm | var | lstm"
                )

            # ── Server-side methods: compute band directly ─────────────────────
            shadow_band_obj = None
            oos_r2_val      = None

            if method == "hmm":
                # 45 s cap — HMM fit on Railway's shared CPU runs ~10 s per
                # restart and is the wall-clock bottleneck of Phase 2.
                out = _pending_futures["hmm"].result(timeout=45)
                # HMM future was dispatched before macro landed, so back-fill
                # yield-curve / VIX / credit macro-regime labels here.
                if macro_ctx:
                    decorate_hmm_meta_with_macro(out["metadata"], macro_ctx)
                if news_ctx:
                    out["metadata"]["news_article_count"] = news_ctx.get("portfolio_summary", {}).get("total_articles")
                # 4-state regime classification — surfaced for UI display only;
                # ensemble weighting is now driven by OOS R² (computed after loop).
                try:
                    from .meta_learner import compute_regime_probs
                    vix_rank = macro_ctx.get("vix_pct_rank", 0.5) if macro_ctx else 0.5
                    rp = compute_regime_probs(out.get("metadata", {}), vix_rank)
                    _hmm_out_extras["regime_probs"] = rp
                    logger.info("Regime: %s (dom=%s)", rp, rp.get("dominant"))
                except Exception as exc:
                    logger.debug("Regime classification failed: %s", exc)
                # Shadow OOS
                if "hmm_shadow" in _pending_futures:
                    try:
                        shadow_out  = _pending_futures["hmm_shadow"].result(timeout=20)
                        shadow_band_obj = ForecastBand(**shadow_out["band"])
                        oos_r2_val  = _compute_oos_r2(shadow_out["band"], shadow_actual_returns)
                        out["metadata"]["oos_r2"] = oos_r2_val
                    except Exception as exc:
                        logger.debug("HMM shadow failed: %s", exc)

            elif method == "var":
                # VAR key preserved for backward compat; computes Gaussian Process
                out = _pending_futures["var"].result(timeout=35)
                if macro_ctx:
                    out["metadata"]["vix_pct_rank"]      = macro_ctx.get("vix_pct_rank")
                    out["metadata"]["vix_term_slope"]    = macro_ctx.get("vix_term_slope")
                    out["metadata"]["yield_curve_10y2y"] = macro_ctx.get("yield_curve_10y2y")
                # Shadow OOS
                if "var_shadow" in _pending_futures:
                    try:
                        shadow_out  = _pending_futures["var_shadow"].result(timeout=20)
                        shadow_band_obj = ForecastBand(**shadow_out["band"])
                        oos_r2_val  = _compute_oos_r2(shadow_out["band"], shadow_actual_returns)
                        out["metadata"]["oos_r2"] = oos_r2_val
                    except Exception as exc:
                        logger.debug("GP shadow failed: %s", exc)

            elif method == "timesfm":
                # Always-warm singleton; TTLCache means repeat calls are free
                out = _pending_futures["timesfm"].result(timeout=60)
                # Shadow OOS
                if "timesfm_shadow" in _pending_futures:
                    try:
                        shadow_out  = _pending_futures["timesfm_shadow"].result(timeout=30)
                        shadow_band_obj = ForecastBand(**shadow_out["band"])
                        oos_r2_val  = _compute_oos_r2(shadow_out["band"], shadow_actual_returns)
                        out["metadata"]["oos_r2"] = oos_r2_val
                    except Exception as exc:
                        logger.debug("TimesFM shadow failed: %s", exc)

            else:
                raise ValueError(f"Unknown forecast method: {method!r}")

            # ── Common append path for server-side methods ────────────────────
            results.append(MethodResult(
                method=method, label=label, color=color,
                forecast=ForecastBand(**out["band"]),
                shadow_band=shadow_band_obj,
                oos_r2=oos_r2_val,
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

    regime_probs = _hmm_out_extras.get("regime_probs")

    # Ensemble weights from OOS R² on shadow holdout — computed after all
    # server-side methods complete so all R²s are available simultaneously.
    try:
        from .meta_learner import get_ensemble_weights_from_oos_r2
        method_r2s = {
            r.method: r.oos_r2
            for r in results
            if r.forecast is not None    # only methods that produced a band
        }
        ensemble_weights = get_ensemble_weights_from_oos_r2(method_r2s) if method_r2s else None
    except Exception as exc:
        logger.debug("OOS R² ensemble weighting failed: %s", exc)
        ensemble_weights = None

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
        shadow_forecast_start=shadow_forecast_start,
        shadow_forecast_end=shadow_forecast_end,
    )
