"""
TimesFM 2.5 forecast provider — official google-research/timesfm package.

CRITICAL: this module uses the `timesfm` package, NOT `transformers`.

Background
----------
The HF transformers integration (TimesFm2_5ModelForPrediction) silently fails
to load most checkpoint weights — the wrapper expects Llama-style separate
q_proj/k_proj/v_proj projections, but Google's actual checkpoint uses the
original combined qkv_proj layout. The result was a randomly-initialised
~200 M-param model serving production traffic for weeks. Verified 2026-05
via load-key inspection (every transformer layer parameter showed as
MISSING from the HF model's perspective).

Switching to `timesfm.TimesFM_2p5_200M_torch.from_pretrained()` loads the
checkpoint cleanly (231 M params, no MISSING/UNEXPECTED keys) and gives
access to the model's native quantile head — eliminating the previous
Gaussian fan synthesis.

Pipeline
--------
  1. load_timesfm()                   — once at startup (load + compile)
  2. forecast_timesfm(returns, …)     — per-request, ~50 ms cached / ~500 ms cold

Inference framing
-----------------
  * Input: raw equity-curve levels  L_t = ∏(1 + r_s)
  * `normalize_inputs=True` lets the model handle scale internally
  * Output: predicted future levels L̂_{T+1..T+H}, plus 10-channel quantiles
  * Convert to cumulative simple-return path: cum_ret_h = (L̂_h − L_T) / L_T
  * Fan bands: native quantile head (no Gaussian synthesis)

References
----------
Das, A. et al. (2024). A decoder-only foundation model for time-series
  forecasting. ICML 2024. https://arxiv.org/abs/2310.10688
Official package: https://github.com/google-research/timesfm
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time

import numpy as np
from cachetools import TTLCache

logger = logging.getLogger(__name__)

MODEL_ID    = "google/timesfm-2.5-200m-pytorch"
CONTEXT_LEN = 512                                         # context fed to the model

# Upper bound of any horizon we'll request. forecast_engine.py uses:
#   main horizon  = 63 trading days (one quarter, from the frontend)
#   shadow horizon = 30 trading days (first half of the 60-day shadow window)
# `max_horizon` in ForecastConfig is an UPPER BOUND — calling forecast(horizon=30)
# on a model compiled with max_horizon=64 works fine. Setting one ceiling lets
# both calls share a single compiled graph (no race-thrashing recompiles).
MAX_HORIZON = 64

_model: object = None                                     # loaded model, or "FAILED"
_load_lock     = threading.Lock()
_cache: TTLCache = TTLCache(maxsize=64, ttl=3_600)        # 1-hour in-process cache


# ── Singleton load ────────────────────────────────────────────────────────────

def load_timesfm() -> None:
    """
    Blocking load called once from the FastAPI lifespan via asyncio.to_thread().
    Thread-safe via double-checked locking. Subsequent callers return immediately
    once _model is set.
    """
    global _model
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        try:
            import timesfm
            from timesfm import ForecastConfig
            t0 = time.time()
            logger.info("TimesFM: loading %s via official timesfm package …", MODEL_ID)
            # torch_compile=False: torch.compile() is brittle on Railway's CPU-only
            # environment and gives modest speedup we don't need (we have a 1-hour
            # in-process cache for repeat queries).
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                MODEL_ID, torch_compile=False,
            )
            n_params = sum(p.numel() for p in model.model.parameters())
            t_loaded = time.time() - t0

            # Pre-compile once with MAX_HORIZON ceiling so per-request forecasts
            # never need to recompile (main h=63 and shadow h=30 share this graph).
            t1 = time.time()
            cfg = ForecastConfig(
                max_context           = CONTEXT_LEN,
                max_horizon           = MAX_HORIZON,
                normalize_inputs      = True,    # internal z-score; we feed raw equity levels
                per_core_batch_size   = 1,       # one series at a time (per-portfolio call)
                force_flip_invariance = True,    # sign-flip TTA — improves directional accuracy
                infer_is_positive     = True,    # equity curves are always positive
                fix_quantile_crossing = True,    # enforce monotonic quantiles in output
            )
            model.compile(cfg)
            _model = model
            logger.info(
                "TimesFM: ready (%d M params, ~%.0f MB fp32) — load %.1fs + compile %.1fs",
                n_params // 1_000_000, n_params * 4 / 1e6,
                t_loaded, time.time() - t1,
            )
        except Exception as exc:
            logger.error("TimesFM: failed to load — %s", exc, exc_info=True)
            _model = "FAILED"


def _get_model():
    """Return the loaded model, or None if it failed / not yet loaded."""
    if _model is None:
        load_timesfm()                  # synchronous fallback (first request only)
    return _model if _model not in (None, "FAILED") else None


# ── Inference ─────────────────────────────────────────────────────────────────

def forecast_timesfm(
    returns: "pd.Series",
    horizon: int,
    last_date: str,
) -> dict:
    """
    Zero-shot probabilistic forecast via TimesFM 2.5 (200 M parameters).

    Pipeline (raw equity-curve framing, linear return conversion):
      1. Build cumulative equity curve  L_t = ∏(1 + r_s) for s ≤ t
      2. Feed the last CONTEXT_LEN level values to TimesFM (normalised internally)
      3. Model returns predicted future levels L̂_{T+1..T+H}  + 10-channel quantiles
      4. Cumulative simple-return path: cum_ret_h = (L̂_{T+h} − L_T) / L_T
      5. Fan bands: native quantile head channels (no Gaussian synthesis)

    Why raw levels:
      * `normalize_inputs=True` means the model z-scores internally and
        denormalises outputs — feeding raw levels works regardless of magnitude.
      * Linear conversion at the end stays bounded (no exp() blowup that
        log-price + exp() would cause on noisy zero-shot output).

    Returns
    -------
    dict: {"band": {dates, p5, p25, p50, p75, p95}, "compute_ms": int, "metadata": dict}
    """
    import pandas as pd
    from app.services.forecast_engine import _forecast_dates

    if len(returns) < 30:
        raise ValueError(
            f"TimesFM requires ≥30 trading days of history, got {len(returns)}. "
            "Try a longer backtest date range."
        )

    # Cache key: SHA-1 of last 252 returns + horizon + date
    key = hashlib.sha1(
        returns.iloc[-252:].values.astype(np.float32).tobytes()
        + f"|{horizon}|{last_date}".encode()
    ).hexdigest()
    if key in _cache:
        return _cache[key]

    model = _get_model()
    if model is None:
        raise RuntimeError(
            "TimesFM model is not available (failed to load at startup — "
            "check Railway logs for 'TimesFM: failed to load')."
        )

    if horizon > MAX_HORIZON:
        raise ValueError(
            f"TimesFM horizon {horizon} exceeds compiled MAX_HORIZON={MAX_HORIZON}. "
            "Bump MAX_HORIZON in timesfm_provider.py and restart."
        )

    t0 = time.time()

    # ── Build raw equity curve ──────────────────────────────────────────────
    levels   = (1.0 + returns).cumprod()
    arr_lvl  = levels.values.astype(np.float32)
    context  = arr_lvl[-CONTEXT_LEN:]
    last_lvl = float(arr_lvl[-1])
    ctx_len  = len(context)

    # ── Forecast ────────────────────────────────────────────────────────────
    # Returns (point, quantiles):
    #   point     : (1, horizon)        — point/mean prediction
    #   quantiles : (1, horizon, 10)    — 10 native quantile channels
    point, quantiles = model.forecast(horizon=horizon, inputs=[context])

    point_levels = point[0]                    # (horizon,)
    q_levels     = quantiles[0]                # (horizon, 10)

    # ── Convert level forecasts → cumulative % returns ──────────────────────
    def _to_cum_pct(arr):
        return ((arr - last_lvl) / last_lvl) * 100.0

    # ── Native quantile bands ───────────────────────────────────────────────
    # Channel layout (TimesFM 2.5 default): [q0.1, q0.2, …, q0.9, mean].
    # Sort along the channel axis as a defence against any layout drift —
    # after `fix_quantile_crossing=True` the model already enforces monotonicity,
    # but sorting is cheap insurance.
    q_sorted = np.sort(q_levels, axis=-1)      # (horizon, 10) ascending
    p10 = _to_cum_pct(q_sorted[:, 0])
    p25 = _to_cum_pct(0.5 * (q_sorted[:, 1] + q_sorted[:, 2]))   # ~q0.25 between q0.2 & q0.3
    p50 = _to_cum_pct(point_levels)                              # use dedicated point head
    p75 = _to_cum_pct(0.5 * (q_sorted[:, 6] + q_sorted[:, 7]))   # ~q0.75 between q0.7 & q0.8
    p90 = _to_cum_pct(q_sorted[:, 8])

    # Extrapolate to p5 / p95 from p10 / p90 using a Gaussian z-ratio.
    # z(0.05)/z(0.10) = -1.645 / -1.282 ≈ 1.283. Symmetric on both tails.
    p5  = p50 - (p50 - p10) * 1.283
    p95 = p50 + (p90 - p50) * 1.283

    dates = _forecast_dates(last_date, horizon)
    band  = {
        "dates": dates,
        "p5":   p5.tolist(),
        "p25":  p25.tolist(),
        "p50":  p50.tolist(),
        "p75":  p75.tolist(),
        "p95":  p95.tolist(),
    }

    result = {
        "band": band,
        "metadata": {
            "is_r2":           None,                    # zero-shot: no training phase
            "model_id":        MODEL_ID,
            "context_len":     ctx_len,
            "input_transform": "raw_level",
            "zero_shot":       True,
            "synthesised_fan": False,                   # native quantile head
            "loader":          "official_timesfm_2p5",  # vs legacy "transformers"
        },
        "compute_ms": int((time.time() - t0) * 1_000),
    }
    _cache[key] = result
    return result
