"""
TimesFM 2.5 forecast provider — self-hosted, always-warm singleton.

Loads google/timesfm-2.5-200m-pytorch once per process at startup via
transformers >= 4.48. Subsequent calls hit the in-process TTLCache (1h)
keyed on a SHA-1 of the last 252 daily returns.

References
----------
Das, A. et al. (2024). A decoder-only foundation model for time-series
  forecasting. ICML 2024. https://arxiv.org/abs/2310.10688
Google Research (2024). TimesFM 2.5 — improved zero-shot time series
  forecasting. https://huggingface.co/google/timesfm-2.5-200m-pytorch
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Optional

import numpy as np
from cachetools import TTLCache

logger = logging.getLogger(__name__)

MODEL_ID    = "google/timesfm-2.5-200m-pytorch"
CONTEXT_LEN = 512                                         # levels fed to the model
_Q_LEVELS   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

_model: object = None          # loaded model, or the sentinel string "FAILED"
_load_lock     = threading.Lock()
_cache: TTLCache = TTLCache(maxsize=64, ttl=3_600)        # 1-hour in-process cache


# ── Singleton load ────────────────────────────────────────────────────────────

def _import_cls():
    """
    Resolve the TimesFM prediction class from transformers >= 4.48.
    Tries the 2.5-specific name first, then the generic name introduced in 4.48.
    Raises ImportError if neither is present.
    """
    for name in ("TimesFm2_5ModelForPrediction", "TimesFmModelForPrediction"):
        try:
            mod = __import__("transformers", fromlist=[name])
            return getattr(mod, name)
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        "TimesFM class not found — install transformers >= 4.48.0 "
        "(pip install 'transformers>=4.48.0')"
    )


def load_timesfm() -> None:
    """
    Blocking load called once from the FastAPI lifespan coroutine via
    asyncio.to_thread().  Thread-safe: only the first caller acquires
    the lock and does the actual download; subsequent callers return
    immediately once _model is set.
    """
    global _model
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:          # double-checked after acquiring lock
            return
        try:
            import torch
            cls = _import_cls()
            logger.info("TimesFM: loading %s …", MODEL_ID)
            t0 = time.time()
            model = cls.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
            model.eval()
            _model = model
            logger.info("TimesFM: ready in %.1fs (RAM +200 MB est.)", time.time() - t0)
        except Exception as exc:
            logger.error("TimesFM: failed to load — %s", exc)
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
    Zero-shot probabilistic forecast via TimesFM 2.5 (200M parameters).

    Pipeline (raw equity-curve framing, linear return conversion):
      1. Build cumulative equity curve  L_t = ∏(1 + r_s) for s ≤ t
      2. Feed the last CONTEXT_LEN level values to TimesFM as past_values
         (no normalisation — TimesFM has internal scaling)
      3. Model returns predicted future levels  L̂_{T+1..T+H}
      4. Cumulative simple-return path: cum_ret_h = (L̂_{T+h} − L_T) / L_T
      5. Fan bands: synthesised in cum-return space as p50 ± z_q · σ · √t,
         where σ is the trailing 252-day realised daily-return vol

    Why raw levels and linear conversion:
      * TimesFM is trained on diverse level series with internal scaling —
        feeding the raw equity curve matches that training shape.
      * Pre-normalising the curve to end at 1.0 fights the model's internal
        scaling and biases it toward mean-of-history forecasts.
      * Log-price framing + exp() conversion is theoretically clean but
        amplifies any model output noise exponentially — small drifts at
        the tail blow up into multi-thousand-percent spikes. Linear
        conversion is bounded and well-behaved.

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

    # Cache key: SHA-1 of last 252 returns as raw bytes + horizon + date
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

    t0 = time.time()

    # ── Build raw equity curve ──────────────────────────────────────────────
    # Cumulative product of (1 + daily returns), no normalisation. TimesFM
    # has internal scaling and handles arbitrary level magnitudes; pre-
    # normalising fights it. Linear conversion at the end keeps the output
    # bounded (no exp() blowup on noisy model predictions).
    levels   = (1.0 + returns).cumprod()                          # equity curve
    arr_lvl  = levels.values.astype(np.float32)
    context  = arr_lvl[-CONTEXT_LEN:]
    last_lvl = float(arr_lvl[-1])                                 # L_T (anchor)
    ctx_len  = len(context)

    import torch
    ctx_tensor = torch.tensor(context).unsqueeze(0)               # (1, ctx_len)

    with torch.no_grad():
        # TimesFm2_5ModelForPrediction uses past_values= (not context=).
        # The HF transformers build does NOT support quantile_levels= kwarg.
        try:
            output = model(past_values=ctx_tensor, prediction_length=horizon)
        except TypeError:
            output = model(past_values=ctx_tensor)

        # HF ModelOutput is dict-like — probe known attribute names.
        def _get(key):
            try:
                v = output[key]
                return v if v is not None else None
            except (KeyError, TypeError):
                return getattr(output, key, None)

        raw_quantiles = (
            _get("quantile_preds")
            or _get("quantile_forecasts")
            or _get("prediction_outputs")
        )
        raw_mean = _get("mean_predictions")

        if raw_quantiles is None and raw_mean is None:
            attrs = sorted(
                k for k in (list(output.keys()) if hasattr(output, "keys") else dir(output))
                if not str(k).startswith("_")
            )
            raise RuntimeError(
                f"TimesFM output format unrecognised — available attrs: {attrs}. "
                "Upgrade transformers or open an issue."
            )

        # Extract central level forecast (median if quantiles, else mean).
        synthesised_fan = True
        pred_levels = None
        if raw_quantiles is not None:
            qarr = raw_quantiles.cpu().numpy() if hasattr(raw_quantiles, "cpu") else np.array(raw_quantiles)
            if qarr.ndim == 3:
                qarr = qarr[0]
            if qarr.ndim == 2 and qarr.shape[-1] == len(_Q_LEVELS):
                pred_levels = qarr[:horizon, 4]          # Q0.5 column = median level
                synthesised_fan = False                  # native quantiles available
        if pred_levels is None:
            arr_mean = raw_mean.cpu().numpy() if hasattr(raw_mean, "cpu") else np.array(raw_mean)
            pred_levels = arr_mean.flatten()[:horizon]

    # ── Convert predicted levels → cumulative simple % return ───────────────
    # cum_ret_h = (L̂_{T+h} − L_T) / L_T  — linear, bounded, no exp() blowup
    cum_p50_pct = ((pred_levels - last_lvl) / last_lvl) * 100.0   # (horizon,)

    # ── Synthesise fan bands in cumulative-return space ─────────────────────
    # σ_daily · √t is the std of cumulative simple return at horizon t under
    # a random-walk-with-drift model. Bands are p50 ± z_q · σ · √t (% units).
    daily_vol   = float(returns.iloc[-252:].std()) if len(returns) >= 5 else 0.01
    t_steps     = np.arange(1, horizon + 1)
    cum_sd_pct  = daily_vol * np.sqrt(t_steps) * 100.0            # in cum-% units

    from scipy.stats import norm as _norm
    z05, z25, z75, z95 = _norm.ppf([0.05, 0.25, 0.75, 0.95])

    def _band_pct(z):
        return (cum_p50_pct + z * cum_sd_pct).tolist()

    dates = _forecast_dates(last_date, horizon)
    band  = {
        "dates": dates,
        "p5":   _band_pct(z05),
        "p25":  _band_pct(z25),
        "p50":  cum_p50_pct.tolist(),
        "p75":  _band_pct(z75),
        "p95":  _band_pct(z95),
    }

    result = {
        "band": band,
        "metadata": {
            "is_r2":           None,   # zero-shot: no training phase
            "model_id":        MODEL_ID,
            "context_len":     ctx_len,
            "input_transform": "raw_level",
            "zero_shot":       True,
            "synthesised_fan": synthesised_fan,
            "sigma_daily":     round(daily_vol, 6),
        },
        "compute_ms": int((time.time() - t0) * 1_000),
    }
    _cache[key] = result
    return result
