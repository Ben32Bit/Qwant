import asyncio
import logging
import anthropic
from fastapi import APIRouter, HTTPException, Request
from app.models.screener import ScreenRequest, RotationBacktestRequest
from app.models.chat import ChatRequest
from app.services.screener_engine import run_screener, run_rotation_backtest
from app.services.screener_ai import call_screener_ai
from app.utils.rate_limit import limiter, AI_LIMITS, CMPT_LIMITS

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/screen/chat")
@limiter.limit(AI_LIMITS)
async def screen_chat(request: Request, chat_request: ChatRequest):
    """
    AI-powered screener: interprets natural language, runs the screen with real
    market data, then feeds results back to Claude for an informed narrative.
    Returns screener results + AI chat text.
    """
    try:
        screen_result, ai_text = await asyncio.to_thread(
            call_screener_ai,
            chat_request.message,
            chat_request.conversation_history,
        )
        return {
            "screen_result": screen_result.model_dump(),
            "ai_response": ai_text,
        }
    except anthropic.APIStatusError as exc:
        if exc.status_code == 529 or "overloaded" in str(exc).lower():
            logger.warning("Anthropic API overloaded (screen): %s", exc)
            raise HTTPException(status_code=503, detail="Anthropic is temporarily overloaded — please try again in a few seconds.")
        if exc.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limit reached — please wait a moment and try again.")
        logger.exception("Anthropic API error (screen): %s", exc)
        raise HTTPException(status_code=502, detail=f"AI service error: {exc.message}")
    except ValueError as exc:
        logger.warning("Screen chat value error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Screen chat unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@router.post("/screen/run")
@limiter.limit(CMPT_LIMITS)
async def run_screen(request: Request, body: ScreenRequest):
    """Direct screen endpoint (no AI — used by manual re-runs)."""
    try:
        result = await asyncio.to_thread(run_screener, body)
        return result.model_dump()
    except Exception as exc:
        logger.exception("Screen run error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@router.post("/screen/backtest")
@limiter.limit(CMPT_LIMITS)
async def backtest_rotation(request: Request, body: RotationBacktestRequest):
    """
    Backtest a momentum rotation strategy from screener results.
    Uses previous-window winners as the holding for the next window (no lookahead).
    """
    try:
        result = await asyncio.to_thread(run_rotation_backtest, body)
        return result.model_dump()
    except Exception as exc:
        logger.exception("Rotation backtest error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")
