import anthropic
import os
import json
from datetime import date
from app.models.screener import ScreenRequest, ScreenerResult

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-sonnet-4-20250514"

SCREENER_SYSTEM_PROMPT = """You are an expert quantitative stock screener assistant.

Your job: understand what the user wants to screen for, then call run_screen with the right parameters.

## Tool: run_screen
Call this ONCE with all the parameters needed. You do NOT need to do research first — just pick a sensible universe.

## Metric Guide
- User says "top returns / best performance / highest return" → metric="return"
- User says "best risk-adjusted / best Sharpe" → metric="sharpe"
- User says "least volatile / lowest risk / most stable" → metric="volatility"
- User says "smallest drawdown / most resilient" → metric="max_drawdown"
- User says "momentum / trend / recent performance" → metric="momentum_3m"

## Universe Guide (when user doesn't specify exact tickers)
- "sectors" / "sector ETFs" → XLK, XLF, XLE, XLV, XLY, XLP, XLU, XLB, XLI, XLRE, XLC
- "tech stocks" / "mega-cap tech" → AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AVGO, ORCL
- "FAANG" → META, AAPL, AMZN, NFLX, GOOGL
- "global / international markets" → VTI, EFA, EEM, VEA, VWO, EWJ, EWZ, FXI
- "asset classes" → SPY, TLT, GLD, VNQ, DBC, HYG, EEM, BTC-USD
- "bonds" → TLT, IEF, SHY, LQD, HYG, AGG, BND, TIPS
- "commodities" → GLD, SLV, USO, DBC, CPER, WEAT, CORN
- "crypto" → BTC-USD, ETH-USD, SOL-USD (yfinance supported)
- Specific tickers mentioned → use exactly those

## Window Frequency
- "quarterly" / "each quarter" → window_freq="quarterly"
- "monthly" / "each month" / "every month" → window_freq="monthly"
- "annually" / "each year" / "year by year" → window_freq="annually"
- "weekly" → window_freq="weekly"
- Default when unspecified: "quarterly"

## top_n
- "top 1" / "best" / "winner" → top_n=1
- "top 3" / "best 3" → top_n=3
- "top 5" → top_n=5
- Default: 3

## Date Range
- If user says "last 3 years" → calculate from today
- If user gives a specific range (e.g. "2020 to 2024") → use that
- Default: last 3 years

## Your Chat Reply (after calling run_screen)
Write exactly 4–5 bullet points. No headers. Be specific and insightful:
- What was screened (universe, period, metric)
- The most notable pattern or winner across windows
- Any consistency or rotation you noticed (e.g., "same sector won 6 of 8 quarters")
- One risk or limitation of this screening approach
- End with: "→ Click **Backtest Rotation Strategy** to test this as a live momentum portfolio"

Keep the reply SHORT and data-aware. Reference actual tickers and windows where possible."""

RUN_SCREEN_TOOL = {
    "name": "run_screen",
    "description": "Screen a universe of assets by a metric across time windows (e.g. best return each quarter).",
    "input_schema": {
        "type": "object",
        "required": ["tickers", "start_date", "end_date", "window_freq", "metric"],
        "properties": {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of ticker symbols to screen (5–20 recommended)",
            },
            "start_date": {"type": "string", "description": "YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            "window_freq": {
                "type": "string",
                "enum": ["weekly", "monthly", "quarterly", "annually"],
                "description": "Time window granularity",
            },
            "metric": {
                "type": "string",
                "enum": ["return", "sharpe", "volatility", "max_drawdown", "momentum_3m"],
                "description": "Metric to rank assets by within each window",
            },
            "top_n": {
                "type": "integer",
                "description": "Number of top assets to highlight per window (default 3)",
            },
            "screen_description": {
                "type": "string",
                "description": "Brief human-readable summary of what was screened",
            },
        },
    },
}


def _default_dates() -> tuple[str, str]:
    end = date.today()
    start = date(end.year - 3, end.month, end.day)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def call_screener_ai(
    message: str,
    conversation_history: list,
) -> tuple[ScreenRequest, str]:
    """
    Single-shot AI call: Claude reads the message and calls run_screen once.
    Returns (ScreenRequest, ai_chat_text).
    """
    start, end = _default_dates()

    messages = [
        {"role": m.role, "content": m.content}
        for m in conversation_history
    ]
    messages.append({"role": "user", "content": message})

    system = [{
        "type": "text",
        "text": SCREENER_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }]

    # Step 1: Claude decides what to screen
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        tools=[RUN_SCREEN_TOOL],
        tool_choice={"type": "required"},   # must call the tool
        messages=messages,
    )

    tool_calls = [b for b in response.content if b.type == "tool_use"]
    if not tool_calls:
        raise ValueError("AI did not produce a screen definition. Please rephrase.")

    tc = tool_calls[0]
    inp = tc.input
    screen_req = ScreenRequest(
        tickers=inp["tickers"],
        start_date=inp.get("start_date", start),
        end_date=inp.get("end_date", end),
        window_freq=inp.get("window_freq", "quarterly"),
        metric=inp.get("metric", "return"),
        top_n=int(inp.get("top_n", 3)),
        screen_description=inp.get("screen_description", ""),
    )

    # Step 2: Claude writes the chat reply
    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tc.id,
            "content": "Screen parameters accepted. Results are being computed now.",
        }],
    })

    final = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=system,
        messages=messages,
    )

    ai_text = " ".join(
        b.text for b in final.content if hasattr(b, "text") and b.text
    ).strip()

    return screen_req, ai_text
