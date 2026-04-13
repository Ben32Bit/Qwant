import asyncio
import logging
from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest
from app.services.unified_ai import call_unified_ai

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/unified/chat")
async def unified_chat(chat_request: ChatRequest):
    """
    Single AI endpoint handling both portfolio construction and asset screening.
    Claude routes the request to the appropriate pipeline based on intent.

    Returns:
        type = "portfolio"     → portfolio + backtest + display_config + ai_response
        type = "screener"      → screen_result + ai_response
        type = "clarification" → ai_response only (Claude asked a clarifying question)
    """
    try:
        # call_unified_ai is synchronous (blocking Anthropic + yfinance calls).
        # Run it in a thread pool so it doesn't stall the async event loop,
        # which would cause Railway's proxy to return a non-JSON 502/504.
        result = await asyncio.to_thread(
            call_unified_ai,
            chat_request.message,
            chat_request.conversation_history,
        )
        return result
    except ValueError as exc:
        logger.warning("Unified chat value error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unified chat unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")
