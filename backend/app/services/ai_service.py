import anthropic
import os
import json
from datetime import date, timedelta
from app.models.chat import ChatMessage, ChatRequest
from app.models.portfolio import PortfolioInput, AssetInput

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are a portfolio construction assistant for a backtesting platform. The user will describe a portfolio or investment strategy in natural language. Your job is to interpret their intent and call the construct_portfolio tool with concrete, investable parameters.

Rules:
- Use real, currently tradable ticker symbols (US-listed ETFs and stocks)
- Weights can be negative (short positions) and can sum to >1.0 (leverage)
- Default date range: last 10 years to today if not specified
- Default rebalancing: quarterly if not specified
- Default benchmark: SPY if not specified
- If the user asks for a strategy (min variance, max sharpe, risk parity, inverse vol, equal weight), set the "strategy" field and provide candidate tickers. The backend will compute optimal weights — you may provide equal weights as placeholders.
- Always include a brief rationale for each holding
- If the request is ambiguous, make reasonable assumptions and state them
- Keep the strategy_summary concise (1-2 sentences)
- After the tool call, briefly explain what you built and any assumptions made"""

PORTFOLIO_TOOL = {
    "name": "construct_portfolio",
    "description": "Construct a portfolio for backtesting from user's natural language description",
    "input_schema": {
        "type": "object",
        "required": ["assets", "start_date", "end_date"],
        "properties": {
            "assets": {
                "type": "array",
                "description": "List of assets with ticker symbols and target weights",
                "items": {
                    "type": "object",
                    "required": ["ticker", "weight"],
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "US-listed ticker symbol (e.g., AAPL, SPY, TLT)"
                        },
                        "weight": {
                            "type": "number",
                            "description": "Target weight. Positive=long, negative=short. Can sum to >1 for leverage."
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Brief reason for including this asset"
                        }
                    }
                }
            },
            "start_date": {
                "type": "string",
                "description": "Backtest start date in YYYY-MM-DD format"
            },
            "end_date": {
                "type": "string",
                "description": "Backtest end date in YYYY-MM-DD format"
            },
            "rebalance_frequency": {
                "type": "string",
                "enum": ["daily", "weekly", "monthly", "quarterly", "annually", "none"],
                "description": "How often to rebalance to target weights. Default: quarterly"
            },
            "benchmark": {
                "type": "string",
                "description": "Benchmark ticker for comparison. Default: SPY"
            },
            "strategy": {
                "type": "string",
                "enum": ["custom", "min_variance", "max_sharpe", "risk_parity", "inverse_vol", "equal_weight"],
                "description": "Portfolio optimization strategy. 'custom' uses the provided weights as-is."
            },
            "strategy_summary": {
                "type": "string",
                "description": "Brief summary of what was constructed and why"
            }
        }
    }
}


def _default_dates() -> tuple[str, str]:
    end = date.today()
    start = date(end.year - 10, end.month, end.day)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def parse_portfolio_from_tool(tool_input: dict) -> PortfolioInput:
    """Convert AI tool call output to PortfolioInput model."""
    start, end = _default_dates()
    return PortfolioInput(
        assets=[
            AssetInput(
                ticker=a["ticker"],
                weight=a["weight"],
                rationale=a.get("rationale"),
            )
            for a in tool_input["assets"]
        ],
        start_date=tool_input.get("start_date", start),
        end_date=tool_input.get("end_date", end),
        rebalance_frequency=tool_input.get("rebalance_frequency", "quarterly"),
        benchmark=tool_input.get("benchmark", "SPY"),
        strategy=tool_input.get("strategy", "custom"),
    ), tool_input.get("strategy_summary", "")


def call_ai(request: ChatRequest) -> tuple[PortfolioInput, str, str]:
    """
    Call Claude with tool use to construct a portfolio from natural language.

    Returns:
        (portfolio_input, ai_response_text, strategy_summary)
    """
    messages = [
        {"role": m.role, "content": m.content}
        for m in request.conversation_history
    ]
    messages.append({"role": "user", "content": request.message})

    # Enable prompt caching for the system prompt
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[PORTFOLIO_TOOL],
        tool_choice={"type": "auto"},
        messages=messages,
    )

    # Extract tool call
    tool_input = None
    text_blocks = []

    for block in response.content:
        if block.type == "tool_use" and block.name == "construct_portfolio":
            tool_input = block.input
        elif block.type == "text":
            text_blocks.append(block.text)

    if tool_input is None:
        raise ValueError("AI did not call construct_portfolio tool. Response: " + " ".join(text_blocks))

    portfolio_input, strategy_summary = parse_portfolio_from_tool(tool_input)

    # Get the follow-up text response (after tool use)
    # If Claude stopped before giving a text response, make a follow-up call
    ai_text = " ".join(text_blocks).strip()
    if not ai_text:
        # Send tool result back to get the final text
        follow_up_messages = messages + [
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": next(
                            b.id for b in response.content if hasattr(b, "type") and b.type == "tool_use"
                        ),
                        "content": "Portfolio constructed successfully. Please provide your analysis.",
                    }
                ],
            },
        ]
        follow_up = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=follow_up_messages,
        )
        ai_text = " ".join(
            b.text for b in follow_up.content if hasattr(b, "text")
        ).strip()

    return portfolio_input, ai_text, strategy_summary
