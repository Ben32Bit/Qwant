import anthropic
import os
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
- "asset classes" → SPY, TLT, GLD, VNQ, DBC, HYG, EEM
- "bonds" → TLT, IEF, SHY, LQD, HYG, AGG, BND
- "commodities" → GLD, SLV, USO, DBC, CPER
- Specific tickers mentioned → use exactly those

## Window Frequency
- "quarterly" / "each quarter" → window_freq="quarterly"
- "monthly" / "each month" → window_freq="monthly"
- "annually" / "each year" → window_freq="annually"
- "weekly" → window_freq="weekly"
- Default when unspecified: "quarterly"

## top_n
- "top 1" / "best" / "winner" → top_n=1
- "top 3" / "best 3" → top_n=3
- "top 5" → top_n=5
- Default: 3

## Date Range
- "last 3 years" → calculate from today
- Specific range e.g. "2020 to 2024" → use that
- Default: last 3 years

## Your Chat Reply
After seeing the REAL results, write exactly 4–5 bullet points. No headers. Be specific:
- What was screened (universe, period, metric)
- The standout winner(s) and how many windows they dominated
- Any notable rotation pattern or regime shift you see
- One key risk or limitation of this approach
- End with: "→ Click **Backtest Rotation Strategy** to test this as a live momentum portfolio"

Reference actual tickers and window counts from the results provided to you."""

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
            },
            "metric": {
                "type": "string",
                "enum": ["return", "sharpe", "volatility", "max_drawdown", "momentum_3m"],
            },
            "top_n": {"type": "integer", "description": "Top N assets per window (default 3)"},
            "screen_description": {"type": "string"},
        },
    },
}


def _default_dates() -> tuple[str, str]:
    end = date.today()
    start = date(end.year - 3, end.month, end.day)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _summarise_results(result: ScreenerResult) -> str:
    """Build a compact text summary of screener results for Claude's narrative step."""
    lines = [
        f"Screen: {result.screen_description or result.metric_label}",
        f"Universe: {', '.join(result.tickers)}",
        f"Windows ({result.window_freq}): {len(result.windows)} total",
        "",
        "Winners per window:",
    ]
    for w in result.windows:
        top = [f"#{r.rank} {r.ticker} ({r.metric_value:+.2%})"
               if result.metric in ("return", "volatility", "max_drawdown", "momentum_3m")
               else f"#{r.rank} {r.ticker} ({r.metric_value:.2f})"
               for r in w.rankings[:result.top_n]]
        lines.append(f"  {w.label}: {', '.join(top)}")

    # Win counts
    from collections import Counter
    win_counts = Counter(w.winner for w in result.windows if w.winner)
    lines.append("")
    lines.append("Win counts: " + ", ".join(
        f"{t}: {c}" for t, c in win_counts.most_common(5)
    ))
    return "\n".join(lines)


def call_screener_ai(
    message: str,
    conversation_history: list,
) -> tuple[ScreenerResult, str]:
    """
    1. Claude calls run_screen tool to choose parameters.
    2. Backend runs the actual screener with real market data.
    3. Real results are passed back to Claude so it writes an informed narrative.

    Returns (ScreenerResult, ai_chat_text).
    """
    from app.services.screener_engine import run_screener

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

    # ── Step 1: Claude chooses screening parameters ────────────────────────────
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        tools=[RUN_SCREEN_TOOL],
        tool_choice={"type": "any"},   # must call a tool; "required" is not a valid value
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

    # ── Step 2: Run the actual screener with real market data ──────────────────
    screen_result = run_screener(screen_req)

    # ── Step 3: Feed real results to Claude so it can write an informed reply ──
    result_summary = _summarise_results(screen_result)

    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tc.id,
            "content": f"Screen complete. Here are the real results:\n\n{result_summary}",
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

    return screen_result, ai_text
