import asyncio
import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.models.forecast import ForecastRequest, ForecastResponse
from app.services.forecast_engine import run_all_forecasts
from app.utils.rate_limit import limiter, FCST_LIMITS

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/forecast", response_model=ForecastResponse)
@limiter.limit(FCST_LIMITS)
async def run_forecast(request: Request, body: ForecastRequest):
    """
    Run probabilistic 3-month portfolio forecasts using up to 6 methods.

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


class PrewarmRequest(BaseModel):
    tickers:  list[str]
    end_date: Optional[str] = None


@router.post("/forecast/prewarm")
async def prewarm_forecast_context(body: PrewarmRequest):
    """
    Speculatively warm the tier-1/tier-2 provider caches for a ticker set.

    Fired by the frontend as soon as a backtest completes, *before* the user
    clicks "Run Forecast". Provider results are keyed on (tickers, date) and
    cached in-process, so when the user does click through, Phase 2 hits warm
    caches instead of 10-25 seconds of sequential network I/O.

    Fire-and-forget from the client's perspective — we return immediately
    while the background tasks run. Failures are swallowed silently since
    this endpoint exists purely as a latency optimisation.
    """
    as_of = body.end_date or date.today().isoformat()
    weights = {t: 1.0 for t in body.tickers}   # equal weighting is only used to gate `available` checks

    async def _warm():
        import concurrent.futures as _cf

        def _macro():
            try:
                from app.services.fred_provider import get_macro_features
                from app.services.vix_provider  import get_vix_features
                get_macro_features(as_of); get_vix_features(as_of)
            except Exception as exc:
                logger.debug("Prewarm macro failed: %s", exc)

        def _insider():
            try:
                from app.services.sec_provider import get_insider_features
                get_insider_features(body.tickers, weights, as_of)
            except Exception as exc:
                logger.debug("Prewarm insider failed: %s", exc)

        def _news():
            try:
                from app.services.news_provider import get_news_context
                get_news_context(body.tickers, weights, as_of)
            except Exception as exc:
                logger.debug("Prewarm news failed: %s", exc)

        def _reddit():
            try:
                from app.services.reddit_provider import get_reddit_context
                get_reddit_context(body.tickers, as_of)
            except Exception as exc:
                logger.debug("Prewarm reddit failed: %s", exc)

        def _edgar():
            try:
                from app.services.edgar_filing_provider import get_edgar_filing_context
                get_edgar_filing_context(body.tickers, weights, as_of)
            except Exception as exc:
                logger.debug("Prewarm edgar failed: %s", exc)

        def _run():
            with _cf.ThreadPoolExecutor(max_workers=5) as pool:
                for fn in (_macro, _insider, _news, _reddit, _edgar):
                    pool.submit(fn)

        await asyncio.to_thread(_run)

    # Schedule the warmup on the event loop and return immediately.
    asyncio.create_task(_warm())
    return {"scheduled": True, "tickers": body.tickers, "as_of": as_of}
