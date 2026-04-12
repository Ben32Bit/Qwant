import anthropic
import os
import json
from datetime import date
from app.models.chat import ChatRequest
from app.models.portfolio import PortfolioInput, AssetInput
from app.services.data_service import get_asset_statistics_for_ai

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-sonnet-4-20250514"
MAX_RESEARCH_ITERATIONS = 5

SYSTEM_PROMPT = """You are an expert portfolio construction assistant backed by real market data.

## Your Tools

**get_asset_statistics** — research tool. Call this BEFORE constructing any data-driven portfolio.
Use it to fetch real returns, volatilities, Sharpe ratios, drawdowns, and correlation matrices for candidate tickers.
You may call it multiple times with different ticker sets or date ranges.

**construct_portfolio** — finalisation tool. Call this ONCE when you have enough data to make informed decisions.

## When to Research First
Always call get_asset_statistics first when the user asks for:
- "most uncorrelated", "lowest correlation", "diversified" strategies
- "best Sharpe", "lowest volatility", "highest return" asset selection
- Any strategy that requires ranking or filtering assets by a quantitative criterion
- Momentum or factor-based selection

For simple explicit portfolios ("60% SPY 40% BND"), you may skip research and go straight to construct_portfolio.

## Rules
- Use real, currently tradable US-listed tickers (ETFs and stocks). Never use ^VIX — use VIXY or VXX instead.
- Weights can be negative (short) and sum to >1.0 (leverage)
- Default date range: last 10 years if not specified
- Default rebalance: quarterly
- Default benchmark: SPY

## Backtest Integrity — Bailey & Lopez de Prado Guidelines
The results include a **Deflated Sharpe Ratio (DSR)** which corrects for overfitting, non-normality, and finite sample bias. Always flag these risks in your narrative:

**Overfitting warnings** (flag if any apply):
- Short backtest period (<3 years): high chance the Sharpe is spurious
- DSR < 0.90: performance likely reflects data-mining rather than edge
- DSR 0.90–0.95: moderate confidence; interpret with caution
- DSR > 0.95: stronger statistical evidence of genuine alpha
- Strategies with many parameters (momentum windows, thresholds) have higher overfitting risk
- In-sample optimisation (e.g. max Sharpe) inflates Sharpe — out-of-sample results will be lower

**Best practices to mention when relevant:**
- Sharpe alone is insufficient — also check DSR, Sortino (penalises downside more), and Calmar (CAGR vs max DD)
- High up-capture with low down-capture is more valuable than raw Sharpe
- Regime sensitivity: strong backtests over a single bull market may fail in different regimes
- Transaction costs and slippage are not modelled — reduce expected returns accordingly for high-turnover strategies
- Benchmark-relative metrics (Alpha, Info Ratio, tracking error) matter more than absolute returns for fund mandates

## Output Format
Write the narrative as **tight bullet points** under short ## headers. Maximum 5 bullets per section. No filler sentences.

Example structure:
```
## What Was Built
- 60% VTI (US equity) + 40% BND (investment-grade bonds), quarterly rebalance
- Benchmark: SPY; 10-year window Jan 2015–Jan 2025

## Key Findings
- CAGR 7.2% vs SPY 13.1% — bonds drag equity returns but reduce vol by 38%
- Sharpe 0.61 / DSR 0.94 — statistically credible, not overfitted
- Max drawdown −18% vs SPY −24% — meaningful downside protection

## Risks & Caveats
- Rate-rise environment (2022) hit both stocks and bonds simultaneously — 60/40 diversification can fail
- No leverage or alternatives — limited upside in risk-on regimes
```

## Display Config
In construct_portfolio, set display_config thoughtfully:
- **sections**: Always include equity_curve, drawdown, metrics_summary. Also add:
  - weight_drift when ≥2 assets
  - correlation_matrix when ≥3 assets and diversification is the goal
  - rolling_metrics when consistency/risk over time matters
  - monthly_heatmap for multi-year strategies
  - full_metrics when benchmark comparison is the focus
- **featured_metrics**: 3–5 metric keys most relevant to this strategy
- **narrative**: Bullet-point markdown per the Output Format above"""

# ── Tool definitions ─────────────────────────────────────────────────────────

GET_ASSET_STATISTICS_TOOL = {
    "name": "get_asset_statistics",
    "description": "Fetch real historical statistics for a list of tickers: annual return, volatility, Sharpe ratio, max drawdown, correlation to benchmark, and pairwise correlation matrix. Use this to make data-driven portfolio decisions.",
    "input_schema": {
        "type": "object",
        "required": ["tickers", "start_date", "end_date"],
        "properties": {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of ticker symbols to analyse (e.g. ['SPY','TLT','GLD','EEM','VNQ'])",
            },
            "start_date": {
                "type": "string",
                "description": "Start date YYYY-MM-DD",
            },
            "end_date": {
                "type": "string",
                "description": "End date YYYY-MM-DD",
            },
            "benchmark": {
                "type": "string",
                "description": "Ticker to compute correlation against. Default: SPY",
            },
        },
    },
}

PORTFOLIO_TOOL = {
    "name": "construct_portfolio",
    "description": "Finalise a portfolio for backtesting. Call this after completing any research.",
    "input_schema": {
        "type": "object",
        "required": ["assets", "start_date", "end_date"],
        "properties": {
            "assets": {
                "type": "array",
                "description": "Assets with ticker and weight",
                "items": {
                    "type": "object",
                    "required": ["ticker", "weight"],
                    "properties": {
                        "ticker": {"type": "string"},
                        "weight": {
                            "type": "number",
                            "description": "Positive=long, negative=short. Can sum >1 for leverage.",
                        },
                        "rationale": {"type": "string"},
                    },
                },
            },
            "start_date": {"type": "string", "description": "YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            "rebalance_frequency": {
                "type": "string",
                "enum": ["daily", "weekly", "monthly", "quarterly", "annually", "none"],
            },
            "benchmark": {"type": "string"},
            "strategy": {
                "type": "string",
                "enum": ["custom", "min_variance", "max_sharpe", "risk_parity", "inverse_vol", "equal_weight"],
            },
            "strategy_summary": {"type": "string"},
            "display_config": {
                "type": "object",
                "description": "Control which sections and metrics the UI shows",
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "equity_curve",
                                "drawdown",
                                "metrics_summary",
                                "full_metrics",
                                "monthly_heatmap",
                                "correlation_matrix",
                                "rolling_metrics",
                                "weight_drift",
                            ],
                        },
                        "description": "Ordered list of sections to display",
                    },
                    "featured_metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metric keys to feature prominently",
                    },
                    "narrative": {
                        "type": "string",
                        "description": "Markdown narrative shown in the results panel explaining the strategy, research findings, and key risks",
                    },
                },
            },
        },
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_dates() -> tuple[str, str]:
    end = date.today()
    start = date(end.year - 10, end.month, end.day)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _parse_portfolio(tool_input: dict) -> tuple[PortfolioInput, str, dict]:
    start, end = _default_dates()
    portfolio = PortfolioInput(
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
    )
    summary = tool_input.get("strategy_summary", "")
    display_config = tool_input.get("display_config", {})
    return portfolio, summary, display_config


def _extract_text(content: list) -> str:
    return " ".join(b.text for b in content if hasattr(b, "text") and b.text).strip()


# ── Main agentic loop ─────────────────────────────────────────────────────────

def call_ai(request: ChatRequest) -> tuple[PortfolioInput, str, str, dict]:
    """
    Agentic loop: Claude can call get_asset_statistics as many times as needed
    before making the final construct_portfolio call.

    Returns:
        (portfolio_input, ai_response_text, strategy_summary, display_config)
    """
    messages = [
        {"role": m.role, "content": m.content}
        for m in request.conversation_history
    ]
    messages.append({"role": "user", "content": request.message})

    tools = [GET_ASSET_STATISTICS_TOOL, PORTFOLIO_TOOL]
    system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

    portfolio_input = None
    strategy_summary = ""
    display_config = {}

    for iteration in range(MAX_RESEARCH_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system,
            tools=tools,
            tool_choice={"type": "auto"},
            messages=messages,
        )

        # Append Claude's response to message history
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [b for b in response.content if b.type == "tool_use"]

        # No tool calls → Claude gave a pure text response (shouldn't happen often)
        if not tool_calls:
            break

        tool_results = []
        construct_called = False

        for tc in tool_calls:
            if tc.name == "get_asset_statistics":
                # Execute the research tool with real market data
                inp = tc.input
                stats = get_asset_statistics_for_ai(
                    tickers=inp["tickers"],
                    start_date=inp["start_date"],
                    end_date=inp["end_date"],
                    benchmark=inp.get("benchmark", "SPY"),
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": json.dumps(stats, default=str),
                })

            elif tc.name == "construct_portfolio":
                portfolio_input, strategy_summary, display_config = _parse_portfolio(tc.input)
                construct_called = True
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": "Portfolio finalised. Backtest is running.",
                })

        # Send tool results back
        messages.append({"role": "user", "content": tool_results})

        if construct_called:
            # Get Claude's final narrative response
            final = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=system,
                messages=messages,
            )
            ai_text = _extract_text(final.content)
            return portfolio_input, ai_text, strategy_summary, display_config

    if portfolio_input is None:
        raise ValueError("AI did not construct a portfolio. Please rephrase your request.")

    return portfolio_input, "", strategy_summary, display_config
