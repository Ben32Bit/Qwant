from fastapi import APIRouter, Request
from app.models.screener import ScreenRequest, RotationBacktestRequest
from app.models.chat import ChatRequest
from app.services.screener_engine import run_screener, run_rotation_backtest
from app.services.screener_ai import call_screener_ai
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/screen/chat")
async def screen_chat(request: ChatRequest, req: Request):
    """
    AI-powered screener: interprets natural language and runs the screen.
    Returns screener results + AI chat text.
    """
    screen_req, ai_text = call_screener_ai(
        message=request.message,
        conversation_history=request.conversation_history,
    )
    result = run_screener(screen_req)
    return {
        "screen_result": result.model_dump(),
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
