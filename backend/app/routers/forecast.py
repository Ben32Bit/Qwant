import asyncio
import logging
from fastapi import APIRouter, HTTPException, Request
from app.models.forecast import ForecastRequest, ForecastResponse
from app.services.forecast_engine import run_all_forecasts
from app.utils.rate_limit import limiter, FCST_LIMITS

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/forecast", response_model=ForecastResponse)
@limiter.limit(FCST_LIMITS)
async def run_forecast(request: Request, body: ForecastRequest):
    """
    Run probabilistic 12-month portfolio forecasts using up to 6 methods.

    Methods: xgboost | nbeats | factor | hmm | var | lstm

    Supports two-phase fetching from the frontend:
      Phase 1: methods=["xgboost","nbeats","factor"]   → ~1-2s
      Phase 2: methods=["hmm","var","lstm"]              → ~10-40s

    Returns a ForecastResponse where each method either has a forecast
    fan band or an error string (no method failure kills the endpoint).
    """
    try:
        result = await asyncio.to_thread(run_all_forecasts, body)
        return result
    except Exception as exc:
        logger.exception("Forecast endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Forecast error: {exc}")
