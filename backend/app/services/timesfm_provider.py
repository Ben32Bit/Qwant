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

    Converts the portfolio daily returns to a normalised equity level series
    (last value = 1.0), feeds the last CONTEXT_LEN values into TimesFM, and
    maps the 9-quantile output to p5/p25/p50/p75/p95 fan bands.

    Quantile index mapping (TimesFM default Q = [0.1, 0.2, ..., 0.9]):
      p5  ← Q0.10  (index 0)
      p25 ← avg(Q0.20, Q0.30)  (indices 1, 2)
      p50 ← Q0.50  (index 4)
      p75 ← avg(Q0.70, Q0.80)  (indices 6, 7)
      p95 ← Q0.90  (index 8)

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

    # ── Build normalised level series ────────────────────────────────────────
    # Reconstruct cumulative equity from returns; normalize so the last
    # observed value = 1.0. TimesFM then predicts future level values;
    # cum_ret[t] = (level[t] - 1.0) * 100 gives cumulative % return.
    levels  = (1.0 + returns).cumprod()
    levels  = levels / float(levels.iloc[-1])           # last value → 1.0
    context = levels.iloc[-CONTEXT_LEN:].values.astype(np.float32)
    ctx_len = len(context)

    import torch
    ctx_tensor = torch.tensor(context).unsqueeze(0)     # shape: (1, ctx_len)

    with torch.no_grad():
        # TimesFm2_5ModelForPrediction uses past_values= (not context=).
        # The HF transformers build does NOT support quantile_levels= kwarg.
        try:
            output = model(
                past_values=ctx_tensor,
                prediction_length=horizon,
            )
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

        synthesised_fan = False
        if raw_quantiles is not None:
            arr = raw_quantiles.cpu().numpy() if hasattr(raw_quantiles, "cpu") else np.array(raw_quantiles)
            if arr.ndim == 3:
                arr = arr[0]
            if arr.ndim == 1 or arr.shape[-1] != len(_Q_LEVELS):
                raw_quantiles = None          # fall through to mean path

        if raw_quantiles is None:
            # HF build returns only mean_predictions — synthesise fan from hist vol
            arr_mean = raw_mean.cpu().numpy() if hasattr(raw_mean, "cpu") else np.array(raw_mean)
            mean_levels = arr_mean.flatten()[:horizon]          # (horizon,)
            daily_vol   = float(returns.iloc[-252:].std()) if len(returns) >= 5 else 0.01
            t_steps     = np.arange(1, horizon + 1)
            cum_vol     = daily_vol * np.sqrt(t_steps)          # growing uncertainty
            from scipy.stats import norm as _norm
            q_probs  = [0.05, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 0.95]
            offsets  = _norm.ppf(q_probs)                       # z-scores for each quantile
            arr = np.zeros((horizon, len(_Q_LEVELS)))
            for col, z in enumerate(offsets):
                arr[:, col] = mean_levels + z * cum_vol         # level space
            synthesised_fan = True

        q = arr                    # (horizon, 9)

    # ── Convert absolute levels → cumulative % return ────────────────────────
    # context ends at 1.0, so predicted level l → return (l - 1) × 100
    cum_ret = (q - 1.0) * 100.0                         # (horizon, 9)

    # ── Build band ───────────────────────────────────────────────────────────
    dates = _forecast_dates(last_date, horizon)
    band  = {
        "dates": dates,
        "p5":   cum_ret[:, 0].tolist(),
        "p25":  ((cum_ret[:, 2] + cum_ret[:, 3]) / 2).tolist(),
        "p50":  cum_ret[:, 4].tolist(),
        "p75":  ((cum_ret[:, 5] + cum_ret[:, 6]) / 2).tolist(),
        "p95":  cum_ret[:, 8].tolist(),
    }

    result = {
        "band": band,
        "metadata": {
            "is_r2":          None,   # zero-shot: no training phase, IS R² not applicable
            "model_id":       MODEL_ID,
            "context_len":    ctx_len,
            "q_levels":       _Q_LEVELS,
            "zero_shot":      True,
            "normalized":     True,
            "synthesised_fan": synthesised_fan,
        },
        "compute_ms": int((time.time() - t0) * 1_000),
    }
    _cache[key] = result
    return result
