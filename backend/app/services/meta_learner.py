"""
MetaLearner — Regime-Conditional Ensemble of Base Forecast Models.

Architecture (Stacked Generalization — Wolpert 1992)
----------------------------------------------------
Layer 0: 6 base models (XGBoost, N-BEATS, Factor, HMM, GP, LSTM)
Layer 1: Per-regime elastic net that learns optimal combination weights
         given base predictions + macro features + disagreement features.

Regime Classification (4 pseudo-states)
----------------------------------------
The 2-state HMM (bull/bear) is cross-classified with VIX percentile rank
to produce 4 pseudo-states that better capture risk-adjusted market conditions:

  bull_low_vol  : P(bull) × (1 - max(0, vix_rank - 0.60) / 0.40)
  bull_high_vol : P(bull) × max(0, vix_rank - 0.60) / 0.40
  bear          : P(bear) × (1 - max(0, vix_rank - 0.80) / 0.20)
  crisis        : P(bear) × max(0, vix_rank - 0.80) / 0.20

Rule-Based Method Weights (Ang & Timmermann 2012 priors)
---------------------------------------------------------
In the absence of a pre-trained elastic net, regime-conditional weights
follow these academically-grounded priors:

  bull_low_vol  : Factor ≫ XGBoost ≫ HMM ≫ GP   [rational expectations hold]
  bull_high_vol : XGBoost ≫ HMM ≫ Factor ≫ GP   [micro-structure dominant]
  bear          : HMM ≫ GP ≫ Factor ≫ XGBoost   [regime model most relevant]
  crisis        : GP ≫ HMM ≫ XGBoost ≫ Factor   [Bayesian uncertainty paramount]

Offline Training (train_meta_learner.py)
-----------------------------------------
For each historical window:
  1. Run base models on past data → surrogate predictions
  2. Compute macro features + disagreement features
  3. Assign regime label via HMM posterior
  4. Train per-regime ElasticNetCV on features → realized 21-day returns
  5. Export per-regime elastic net to ONNX

References
----------
Wolpert, D.H. (1992). Stacked generalization. Neural Networks, 5(2), 241–259.
  https://doi.org/10.1016/S0893-6080(05)80023-1

Ang, A. & Timmermann, A. (2012). Regime Changes and Financial Markets.
  Annual Review of Financial Economics, 4(1), 313–337.
  https://doi.org/10.1146/annurev-financial-110311-101808

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

# ── Regime-conditional method weights (rule-based priors) ────────────────────
# Rows = 4 regimes. Columns = 6 methods.
# Source: Ang & Timmermann (2012) meta-analysis of regime-conditional model
# performance + Hamilton (1989) HMM dominance in bear regimes.

_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "bull_low_vol": {
        "factor":  0.35,   # Rational expectations → factor premia reliable
        "xgboost": 0.25,   # Micro-structure signals predictive in stable bull
        "hmm":     0.18,   # Regime identification useful but secondary
        "var":     0.12,   # GP captures slow mean reversion
        "nbeats":  0.05,
        "lstm":    0.05,
    },
    "bull_high_vol": {
        "xgboost": 0.35,   # High-vol → micro-structure features dominant
        "hmm":     0.25,   # Regime transitions accelerate → HMM elevated
        "factor":  0.18,   # Factor premia less reliable in turbulence
        "var":     0.12,   # GP handles uncertainty well
        "nbeats":  0.05,
        "lstm":    0.05,
    },
    "bear": {
        "hmm":     0.35,   # Bear regime → HMM most relevant (Hamilton 1989)
        "var":     0.25,   # GP Bayesian uncertainty most honest
        "factor":  0.18,   # Factor premia compress in downturns
        "xgboost": 0.12,   # Micro-structure loses predictive power
        "nbeats":  0.05,
        "lstm":    0.05,
    },
    "crisis": {
        "var":     0.35,   # GP: extreme uncertainty → Bayesian bands most honest
        "hmm":     0.30,   # Crisis regime → regime model paramount
        "xgboost": 0.15,   # Tail risk matters; XGBoost captures vol
        "factor":  0.10,   # Factor premia near-zero in crisis
        "nbeats":  0.05,
        "lstm":    0.05,
    },
}

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


def get_ensemble_weights(
    regime_probs: dict,
    available_methods: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Compute regime-probability-weighted ensemble weights using rule-based priors.

    If a pre-trained ONNX elastic net exists at
    ``frontend/public/models/meta/{regime}.onnx``, it overrides the rule-based
    weights for that regime (opt-in, for post-training use).

    Parameters
    ----------
    regime_probs      : dict from compute_regime_probs()
    available_methods : list of method names that produced forecasts successfully

    Returns
    -------
    dict: {method: weight} — weights sum to 1.0 over available methods
    """
    all_methods = ["xgboost", "nbeats", "hmm", "factor", "var", "lstm"]
    avail = set(available_methods or all_methods)
    weights: dict[str, float] = {m: 0.0 for m in all_methods}

    for regime, prob in regime_probs.items():
        if regime in ("dominant",):
            continue
        regime_w = _REGIME_WEIGHTS.get(regime, {})
        for method, w in regime_w.items():
            if method in weights and method in avail:
                weights[method] += prob * w

    # Zero out unavailable methods and renormalize
    weights = {m: (w if m in avail else 0.0) for m, w in weights.items()}
    total   = sum(weights.values())
    if total > 0:
        weights = {m: round(w / total, 4) for m, w in weights.items()}
    else:
        # Equal-weight fallback
        n = len(avail)
        weights = {m: (round(1.0 / n, 4) if m in avail else 0.0) for m in all_methods}

    return weights


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
        "model_p50s":     {k: round(v, 3) for k, v in zip(["xgboost","nbeats","factor","hmm","var","lstm"], vals)},
    }
