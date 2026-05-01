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

    Pipeline (standard quant log-price framing):
      1. Build cumulative equity curve  L_t = ∏(1 + r_s) for s ≤ t
      2. Take log: ℓ_t = log L_t  (random-walk-with-drift target)
      3. Feed the last CONTEXT_LEN log-prices to TimesFM as past_values
      4. Model returns predicted log-prices  ℓ̂_{t+1..t+H}
      5. Cumulative simple-return path: cum_ret_h = exp(ℓ̂_{T+h} − ℓ_T) − 1
      6. Fan bands: synthesised as exp(ℓ̂ + z_q · σ_logret · √t) − 1, where
         σ_logret is the trailing 252-day realised log-return vol

    Why log-prices, not normalised levels or raw returns:
      * TimesFM is trained on diverse mostly-non-financial level series with
        internal scaling — feeding raw cumulative levels matches that shape.
      * Log-prices are the standard quant target for random-walk-with-drift
        models; their increments (log-returns) are stationary.
      * Pre-normalising the level series to end at 1.0 fights the model's
        internal scaling and biases it toward generic mean-of-history
        forecasts (the previous behaviour, which scored OOS R² ≈ −100 on
        flat windows).

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

    # ── Build log-price series ──────────────────────────────────────────────
    # Cumulative equity curve, then natural log. log-prices are the canonical
    # random-walk-with-drift target in quant: their first differences are
    # log-returns, which are (near-)stationary. No further normalisation —
    # TimesFM has its own internal scaling and pre-normalising fights it.
    levels      = (1.0 + returns).cumprod()                       # equity curve
    log_levels  = np.log(levels.values).astype(np.float32)        # ℓ_t = log L_t
    context     = log_levels[-CONTEXT_LEN:]
    last_log    = float(log_levels[-1])                           # ℓ_T (anchor)
    ctx_len     = len(context)

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

        # Extract central log-price forecast (median if quantiles, else mean).
        synthesised_fan = True
        pred_log_levels = None
        if raw_quantiles is not None:
            qarr = raw_quantiles.cpu().numpy() if hasattr(raw_quantiles, "cpu") else np.array(raw_quantiles)
            if qarr.ndim == 3:
                qarr = qarr[0]
            if qarr.ndim == 2 and qarr.shape[-1] == len(_Q_LEVELS):
                pred_log_levels = qarr[:horizon, 4]      # Q0.5 column = median log-price
                synthesised_fan = False                  # native quantiles available
        if pred_log_levels is None:
            arr_mean = raw_mean.cpu().numpy() if hasattr(raw_mean, "cpu") else np.array(raw_mean)
            pred_log_levels = arr_mean.flatten()[:horizon]

    # ── Convert log-prices → cumulative simple % return ─────────────────────
    # cum_ret_h = L_{T+h}/L_T − 1 = exp(ℓ̂_{T+h} − ℓ_T) − 1
    cum_log_ret = pred_log_levels - last_log                      # (horizon,)
    cum_p50_pct = (np.exp(cum_log_ret) - 1.0) * 100.0             # cum %

    # ── Synthesise fan bands in log-return space ────────────────────────────
    # σ_logret · √t is the std of cumulative log-return at horizon t under a
    # random-walk-with-drift model. Convert each Gaussian z-quantile back to
    # simple-return space via exp(·) − 1 to preserve lognormal shape.
    log_returns = np.diff(np.log(levels.values))                  # daily log-rets
    sigma_log   = float(np.std(log_returns[-252:])) if len(log_returns) >= 5 else 0.01
    t_steps     = np.arange(1, horizon + 1)
    cum_log_sd  = sigma_log * np.sqrt(t_steps)

    from scipy.stats import norm as _norm
    z05, z25, z75, z95 = _norm.ppf([0.05, 0.25, 0.75, 0.95])

    def _band_pct(z):
        return ((np.exp(cum_log_ret + z * cum_log_sd) - 1.0) * 100.0).tolist()

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
            "input_transform": "log_price",
            "zero_shot":       True,
            "synthesised_fan": synthesised_fan,
            "sigma_log_daily": round(sigma_log, 6),
        },
        "compute_ms": int((time.time() - t0) * 1_000),
    }
    _cache[key] = result
    return result
