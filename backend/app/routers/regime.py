"""
GET/POST /api/regime — Current market regime detection.

Returns 4-state regime probabilities derived from the 2-state HMM
(bull/bear) cross-classified with the live VIX percentile rank.

Designed for on-demand queries: accepts a portfolio equity curve, runs
the HMM on the last 252 trading days, cross-classifies with VIX, and
returns regime probabilities + ensemble weight recommendations.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class RegimeRequest(BaseModel):
    equity_curve: list[dict]           # [{date: str, value: float}, ...]
    end_date: Optional[str] = None     # YYYY-MM-DD; inferred from equity_curve if omitted


@router.post("/regime/current")
async def get_current_regime(req: RegimeRequest, request: Request):
    """
    Detect current market regime from a portfolio equity curve.

    Returns
    -------
    {
      regime_probabilities: {bull_low_vol, bull_high_vol, bear, crisis},
      dominant_regime: str,
      ensemble_weights: {method: weight},
      hmm_metadata: {...},
      macro_context: {...},
      compute_ms: int
    }
    """
    import time
    t0 = time.time()

    try:
        import pandas as pd
        import numpy as np
        from app.services.forecast_engine import _equity_to_returns, forecast_hmm
        from app.services.vix_provider     import get_vix_features
        from app.services.meta_learner     import compute_regime_probs, get_ensemble_weights

        end_date = req.end_date or req.equity_curve[-1]["date"]
        returns  = _equity_to_returns(req.equity_curve)

        # Run HMM (2-state)
        hmm_out = forecast_hmm(returns, horizon=21, n_paths=200, last_date=end_date)
        hmm_meta = hmm_out.get("metadata", {})

        # Live VIX rank
        vix_feats   = get_vix_features(end_date)
        vix_pct_rank = float(vix_feats.get("vix_pct_rank", 0.5))

        # 4-state regime probabilities
        regime_probs = compute_regime_probs(hmm_meta, vix_pct_rank)
        dominant     = regime_probs.get("dominant", "bull_low_vol")

        # Ensemble weights
        weights = get_ensemble_weights(regime_probs)

        return {
            "regime_probabilities": {k: v for k, v in regime_probs.items() if k != "dominant"},
            "dominant_regime":      dominant,
            "ensemble_weights":     weights,
            "hmm_metadata":         hmm_meta,
            "vix_pct_rank":         round(vix_pct_rank, 3),
            "compute_ms":           int((time.time() - t0) * 1000),
        }

    except Exception as e:
        logger.exception("Regime detection failed")
        return {"error": str(e), "compute_ms": int((time.time() - t0) * 1000)}
