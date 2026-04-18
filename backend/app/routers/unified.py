import asyncio
import logging
import anthropic
from fastapi import APIRouter, HTTPException, Request
from app.models.chat import ChatRequest
from app.services.unified_ai import call_unified_ai
from app.utils.rate_limit import limiter, AI_LIMITS

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/unified/chat")
@limiter.limit(AI_LIMITS)
async def unified_chat(request: Request, chat_request: ChatRequest):
    """
    Single AI endpoint handling both portfolio construction and asset screening.
    Claude routes the request to the appropriate pipeline based on intent.

    Returns:
        type = "portfolio"     → portfolio + backtest + display_config + ai_response
        type = "screener"      → screen_result + ai_response
        type = "clarification" → ai_response only (Claude asked a clarifying question)
    """
    try:
        result = await asyncio.to_thread(
            call_unified_ai,
            chat_request.message,
            chat_request.conversation_history,
        )
        return result
    except anthropic.APIStatusError as exc:
        if exc.status_code == 529 or "overloaded" in str(exc).lower():
            logger.warning("Anthropic API overloaded: %s", exc)
            raise HTTPException(status_code=503, detail="Anthropic is temporarily overloaded — please try again in a few seconds.")
        if exc.status_code == 429:
            logger.warning("Anthropic rate limit: %s", exc)
            raise HTTPException(status_code=429, detail="Rate limit reached — please wait a moment and try again.")
        logger.exception("Anthropic API error: %s", exc)
        raise HTTPException(status_code=502, detail=f"AI service error: {exc.message}")
    except ValueError as exc:
        logger.warning("Unified chat value error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unified chat unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")
