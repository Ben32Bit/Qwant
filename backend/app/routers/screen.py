from fastapi import APIRouter, Request
from app.models.screener import ScreenRequest, RotationBacktestRequest
from app.models.chat import ChatRequest
from app.services.screener_engine import run_screener, run_rotation_backtest
from app.services.screener_ai import call_screener_ai
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/screen/chat")
async def screen_chat(chat_request: ChatRequest):
    """
    AI-powered screener: interprets natural language, runs the screen with real
    market data, then feeds results back to Claude for an informed narrative.
    Returns screener results + AI chat text.
    """
    screen_result, ai_text = call_screener_ai(
        message=chat_request.message,
        conversation_history=chat_request.conversation_history,
    )
    return {
        "screen_result": screen_result.model_dump(),
        "ai_response": ai_text,
    }


@router.post("/screen/run")
async def run_screen(request: ScreenRequest):
    """Direct screen endpoint (no AI — used by manual re-runs)."""
    result = run_screener(request)
    return result.model_dump()


@router.post("/screen/backtest")
async def backtest_rotation(request: RotationBacktestRequest):
    """
    Backtest a momentum rotation strategy from screener results.
    Uses previous-window winners as the holding for the next window (no lookahead).
    """
    result = run_rotation_backtest(request)
    return result.model_dump()
