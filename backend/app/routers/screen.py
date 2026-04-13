import asyncio
import logging
from fastapi import APIRouter, HTTPException, Request
from app.models.screener import ScreenRequest, RotationBacktestRequest
from app.models.chat import ChatRequest
from app.services.screener_engine import run_screener, run_rotation_backtest
from app.services.screener_ai import call_screener_ai

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/screen/chat")
async def screen_chat(chat_request: ChatRequest):
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
    except ValueError as exc:
        logger.warning("Screen chat value error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Screen chat unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@router.post("/screen/run")
async def run_screen(request: ScreenRequest):
    """Direct screen endpoint (no AI — used by manual re-runs)."""
    try:
        result = await asyncio.to_thread(run_screener, request)
        return result.model_dump()
    except Exception as exc:
        logger.exception("Screen run error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@router.post("/screen/backtest")
async def backtest_rotation(request: RotationBacktestRequest):
    """
    Backtest a momentum rotation strategy from screener results.
    Uses previous-window winners as the holding for the next window (no lookahead).
    """
    try:
        result = await asyncio.to_thread(run_rotation_backtest, request)
        return result.model_dump()
    except Exception as exc:
        logger.exception("Rotation backtest error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")
