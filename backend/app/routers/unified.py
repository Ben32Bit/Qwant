from fastapi import APIRouter
from app.models.chat import ChatRequest
from app.services.unified_ai import call_unified_ai
import logging

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
    result = call_unified_ai(
        message=chat_request.message,
        conversation_history=chat_request.conversation_history,
    )
    return result
