"""
MetaLearner — OOS R²-Weighted Ensemble of Base Forecast Models.

Architecture (Stacked Generalization — Wolpert 1992)
----------------------------------------------------
Layer 0: 5 base models (N-BEATS, HMM, GP, LSTM, TimesFM 2.5)
Layer 1: Ensemble weights proportional to OOS R² on a 30-day shadow holdout,
         with a 30% floor for TimesFM (zero-shot foundation model stability).

         Negative OOS R² is clipped to 0 (a model worse than the naive mean
         contributes nothing). If all methods have zero R², equal weight is
         used as a fallback.

Regime Classification (4 pseudo-states)
----------------------------------------
HMM bull/bear probability is cross-classified with VIX percentile rank to
produce 4 pseudo-states. These are surfaced for UI display (regime donut,
ScenarioPanel) but do NOT drive ensemble weighting — that role is now filled
by OOS R² measured on the shadow holdout window.

References
----------
Wolpert, D.H. (1992). Stacked generalization. Neural Networks, 5(2), 241–259.
  https://doi.org/10.1016/S0893-6080(05)80023-1

Krogh, A. & Vedelsby, J. (1995). Neural Network Ensembles, Cross Validation,
  and Active Learning. Advances in Neural Information Processing Systems, 8.

Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and Scalable
  Predictive Uncertainty Estimation Using Deep Ensembles. NeurIPS 30.

Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary
  time series and the business cycle. Econometrica, 57(2), 357–384.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

REGIME_COLORS = {
    "bull_low_vol":  "#00d4aa",   # green
    "bull_high_vol": "#ffd43b",   # amber
    "bear":          "#ff6b35",   # orange
    "crisis":        "#ff4757",   # red
}

REGIME_LABELS = {
    "bull_low_vol":  "Bull / Low Vol",
    "bull_high_vol": "Bull / High Vol",
    "bear":          "Bear",
    "crisis":        "Crisis",
}


# ── Public interface ──────────────────────────────────────────────────────────

def compute_regime_probs(hmm_metadata: dict, vix_pct_rank: float = 0.5) -> dict:
    """
    Map 2-state HMM output to 4 pseudo-states using VIX percentile rank.

    Parameters
    ----------
    hmm_metadata  : dict from forecast_hmm() — requires 'current_bull_prob'
    vix_pct_rank  : float in [0, 1] — VIX percentile rank vs 5-year history

    Returns
    -------
    dict: {bull_low_vol: float, bull_high_vol: float, bear: float, crisis: float}
    """
    bull_p = float(hmm_metadata.get("current_bull_prob", 0.5))
    bull_p = max(0.0, min(1.0, bull_p))
    bear_p = 1.0 - bull_p

    # Bull split at VIX 60th percentile
    if vix_pct_rank <= 0.60:
        bull_low_vol  = bull_p
        bull_high_vol = 0.0
    else:
        frac          = min((vix_pct_rank - 0.60) / 0.40, 1.0)
        bull_high_vol = bull_p * frac
        bull_low_vol  = bull_p * (1.0 - frac)

    # Bear split at VIX 80th percentile
    if vix_pct_rank <= 0.80:
        bear_normal = bear_p
        crisis      = 0.0
    else:
        frac        = min((vix_pct_rank - 0.80) / 0.20, 1.0)
        crisis      = bear_p * frac
        bear_normal = bear_p * (1.0 - frac)

    total = bull_low_vol + bull_high_vol + bear_normal + crisis
    if total == 0:
        total = 1.0

    probs = {
        "bull_low_vol":  round(bull_low_vol  / total, 4),
        "bull_high_vol": round(bull_high_vol / total, 4),
        "bear":          round(bear_normal   / total, 4),
        "crisis":        round(crisis        / total, 4),
    }
    dominant = max(probs, key=probs.get)
    return {**probs, "dominant": dominant}


def get_ensemble_weights_from_oos_r2(
    method_r2s: dict[str, Optional[float]],
) -> dict[str, float]:
    """
    Weight ensemble methods proportional to their OOS R² on the shadow holdout.

    TimesFM gets a 30% floor (zero-shot foundation model stability — its R² can
    vary wildly on short windows without warranting a low allocation).

    Algorithm
    ---------
    1. Clip negative R² to 0 — a model worse than the naive mean contributes nothing.
    2. Normalize clipped values to sum to 1.
    3. Apply TimesFM 30% floor: if timesfm weight < 0.30, redistribute the
       deficit proportionally from the other methods.
    4. Fallback to equal weight when all clipped R²s are 0.

    Parameters
    ----------
    method_r2s : {method: oos_r2} — None values are treated as 0.

    Returns
    -------
    dict: {method: weight} — weights sum to 1.0 over all included methods
    """
    TFM_FLOOR = 0.30

    clipped = {m: max(0.0, r2 or 0.0) for m, r2 in method_r2s.items()}
    total   = sum(clipped.values())

    if total <= 0:
        n = len(method_r2s)
        return {m: round(1.0 / n, 4) if n > 0 else 0.0 for m in method_r2s}

    weights = {m: w / total for m, w in clipped.items()}

    # TimesFM floor
    if "timesfm" in weights and weights["timesfm"] < TFM_FLOOR:
        deficit = TFM_FLOOR - weights["timesfm"]
        weights["timesfm"] = TFM_FLOOR
        others       = {m: w for m, w in weights.items() if m != "timesfm"}
        other_total  = sum(others.values())
        if other_total > 0:
            scale = (1.0 - TFM_FLOOR) / other_total
            for m in others:
                weights[m] = others[m] * scale

    return {m: round(w, 4) for m, w in weights.items()}


def compute_disagreement(method_p50s: dict[str, list[float]], horizon_idx: int = 20) -> dict:
    """
    Compute ensemble disagreement features from base model 21-day p50 forecasts.

    Parameters
    ----------
    method_p50s  : {method: [p50 at each forecast step]} — cumulative % returns
    horizon_idx  : index to evaluate (default 20 = 21st day)

    Returns
    -------
    dict: {level, spread, variance, pairwise_corr, n_models}

    References: Krogh & Vedelsby (1995), Lakshminarayanan et al. (2017)
    """
    vals = []
    for method, p50 in method_p50s.items():
        if p50 and len(p50) > horizon_idx:
            vals.append(float(p50[horizon_idx]))

    n = len(vals)
    if n < 2:
        return {"level": "low", "spread": 0.0, "variance": 0.0, "pairwise_corr": 1.0, "n_models": n}

    arr      = np.array(vals)
    variance = float(np.var(arr))
    spread   = float(np.ptp(arr))   # max - min

    # Pairwise correlation: for scalar values this is std/mean
    mean = float(np.mean(arr))
    cv   = float(np.std(arr) / (abs(mean) + 1e-9))   # coefficient of variation

    # Thresholds calibrated to typical fan-band returns (% cumulative)
    if variance > 2.5 or spread > 8.0:
        level = "high"
    elif variance > 0.5 or spread > 3.0:
        level = "med"
    else:
        level = "low"

    return {
        "level":          level,
        "spread":         round(spread, 3),
        "variance":       round(variance, 4),
        "coeff_var":      round(cv, 3),
        "n_models":       n,
        "model_p50s":     {k: round(v, 3) for k, v in zip(["nbeats","hmm","var","lstm","timesfm"], vals)},
    }
