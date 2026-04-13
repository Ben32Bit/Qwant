import anthropic
import os
import json
from datetime import date
from app.models.portfolio import PortfolioInput, AssetInput
from app.models.screener import ScreenRequest
from app.services.data_service import get_asset_statistics_for_ai
from app.services.screener_engine import run_screener
from app.services.screener_ai import _summarise_results

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-sonnet-4-20250514"
MAX_RESEARCH_ITERATIONS = 6   # portfolio path can use up to 5 research calls + 1 final

# ── System prompt ─────────────────────────────────────────────────────────────

UNIFIED_SYSTEM_PROMPT = """You are Qwant — an AI investment analyst. You have two capabilities and must choose the right one for each request.

---

## ROUTING GUIDE — read intent first, then call the right tool

### → SCREEN  (call `run_screen`)
User wants to know **which asset ranked best in each time window** — retroactive period-by-period comparison.

**Signal phrases:** "top", "best", "which performed", "which led", "rank", "screen", "each quarter / each month / each year", "over rolling windows", "who won each period"

**Examples:**
- "Which sector ETF had the best return each quarter 2022–2025?"
- "Best Sharpe ratio among FAANG stocks by year since 2018"
- "Top commodity each quarter: GLD SLV USO DBC"
- "Least volatile asset class each month in 2023"

### → BUILD PORTFOLIO  (call `construct_portfolio`, optionally after `get_asset_statistics`)
User wants to **build, allocate, and backtest a portfolio** — weights, strategy, equity curve.

**Signal phrases:** "portfolio", "invest", "allocate", "build", "backtest", weights or percentages, strategy names (60/40, risk parity, all-weather, FAANG, max Sharpe, min variance), "vs SPY", "vs benchmark"

**Examples:**
- "60/40 US stocks and bonds over 10 years"
- "All-weather portfolio since 2010"
- "Max Sharpe from SPY, TLT, GLD, EEM"
- "Equal weight FAANG vs SPY since 2018"

### When ambiguous
Ask exactly ONE short clarifying question. Example: *"Are you looking to screen which asset led each period, or build a portfolio to backtest?"*

---

## Tool: run_screen
Call this ONCE — no prior research needed. Pick parameters directly from the request.

### Metric Guide
- "top returns / best performance" → metric="return"
- "best risk-adjusted / best Sharpe" → metric="sharpe"
- "least volatile / most stable / lowest risk" → metric="volatility"
- "smallest drawdown / most resilient" → metric="max_drawdown"
- "momentum / recent trend" → metric="momentum_3m"

### Universe Guide
- "sectors / sector ETFs" → XLK, XLF, XLE, XLV, XLY, XLP, XLU, XLB, XLI, XLRE, XLC
- "mega-cap tech / FAANG" → META, AAPL, AMZN, NFLX, GOOGL (FAANG) or add MSFT, NVDA, TSLA, AVGO
- "global / international markets" → VTI, EFA, EEM, VEA, VWO, EWJ, EWZ, FXI
- "asset classes" → SPY, TLT, GLD, VNQ, DBC, HYG, EEM
- "bonds" → TLT, IEF, SHY, LQD, HYG, AGG, BND
- "commodities" → GLD, SLV, USO, DBC, CPER
- Explicit tickers → use exactly those

### Window Frequency
- "quarterly" / "each quarter" → window_freq="quarterly"
- "monthly" / "each month" → window_freq="monthly"
- "annually" / "each year" → window_freq="annually"
- "weekly" → window_freq="weekly"
- Default: "quarterly"

### top_n
- "top 1 / best / winner" → top_n=1
- "top 3" → top_n=3  |  "top 5" → top_n=5
- Default: 3

### Date Range
- Default if unspecified: last 3 years

---

## Tool: get_asset_statistics
Research tool for portfolio construction. Call BEFORE `construct_portfolio` when the user asks for:
- "most uncorrelated", "lowest correlation", "diversified" strategies
- "best Sharpe", "highest return", "lowest volatility" asset selection
- Momentum or factor-based weight decisions

Skip this for simple explicit portfolios ("60% SPY 40% BND") — go straight to `construct_portfolio`.

---

## Tool: construct_portfolio
Finalise a portfolio. Call once after research (if needed).

### Rules
- Real, US-listed tickers only. Never ^VIX — use VIXY or VXX instead.
- Weights can be negative (short) and sum to >1.0 (leverage)
- Default range: last 10 years | Default rebalance: quarterly | Default benchmark: SPY

### display_config
Set thoughtfully in the tool call:
- **sections**: Always include equity_curve, drawdown, metrics_summary. Add:
  - weight_drift when ≥2 assets
  - correlation_matrix when ≥3 assets and diversification matters
  - rolling_metrics when consistency over time is the focus
  - monthly_heatmap for multi-year strategies
  - full_metrics when benchmark comparison is the focus
- **featured_metrics**: 3–5 keys most relevant to the strategy
- **narrative**: Full markdown analysis (see Output Format below)

---

## Backtest Integrity — Bailey & Lopez de Prado
Results include a **Deflated Sharpe Ratio (DSR)** that corrects for overfitting and non-normality. Flag these risks when relevant:

- DSR > 0.95 → stronger evidence of genuine alpha
- DSR 0.90–0.95 → moderate confidence, interpret with caution
- DSR < 0.90 → performance may reflect data-mining, not real edge
- Short backtest (<3 years) → high chance the Sharpe is spurious
- In-sample optimisation (max Sharpe, min variance) inflates reported Sharpe — real out-of-sample will be lower
- High turnover strategies: transaction costs and slippage not modelled — reduce expected returns

---

## Output Format

### After run_screen + real results → 4–5 bullets in chat
- What was screened (universe, period, metric)
- The standout winner(s) and how many windows they dominated
- Any notable rotation pattern or regime shift
- One key risk or limitation
- End with: "→ Click **Backtest Rotation Strategy** to test this as a live momentum portfolio"

Reference actual tickers and window counts from the results provided.

### After construct_portfolio → TWO separate texts

**display_config.narrative** (results panel — detailed markdown):
```
## What Was Built
- [tickers, weights, strategy, rebalance, date range]

## Key Findings
- [CAGR, Sharpe, DSR commentary — be specific with numbers]
- [Drawdown, risk metrics]
- [vs benchmark if relevant]

## Risks & Caveats
- [Bailey/overfitting warnings if applicable]
- [Regime sensitivity, concentration, etc.]

## Suggested Next Steps
- [1–2 actionable refinements]
```

**Final chat reply** (chat panel — 3–5 bullets, no headers, concise):
- What was built + headline result
- One statistical insight (DSR or Sharpe vs benchmark)
- 1–2 suggested next steps"""

# ── Tool definitions ──────────────────────────────────────────────────────────

GET_ASSET_STATISTICS_TOOL = {
    "name": "get_asset_statistics",
    "description": "Fetch real historical statistics for a list of tickers: annual return, volatility, Sharpe ratio, max drawdown, correlation to benchmark, and pairwise correlation matrix. Use before construct_portfolio for data-driven portfolios.",
    "input_schema": {
        "type": "object",
        "required": ["tickers", "start_date", "end_date"],
        "properties": {
            "tickers": {"type": "array", "items": {"type": "string"}},
            "start_date": {"type": "string", "description": "YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            "benchmark": {"type": "string", "description": "Correlation benchmark. Default: SPY"},
        },
    },
}

PORTFOLIO_TOOL = {
    "name": "construct_portfolio",
    "description": "Finalise a portfolio for backtesting. Call after any needed research.",
    "input_schema": {
        "type": "object",
        "required": ["assets", "start_date", "end_date"],
        "properties": {
            "assets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["ticker", "weight"],
                    "properties": {
                        "ticker": {"type": "string"},
                        "weight": {"type": "number", "description": "Positive=long, negative=short. Can sum >1 for leverage."},
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
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["equity_curve", "drawdown", "metrics_summary", "full_metrics",
                                     "monthly_heatmap", "correlation_matrix", "rolling_metrics", "weight_drift"],
                        },
                    },
                    "featured_metrics": {"type": "array", "items": {"type": "string"}},
                    "narrative": {"type": "string", "description": "Markdown analysis shown in results panel"},
                },
            },
        },
    },
}

RUN_SCREEN_TOOL = {
    "name": "run_screen",
    "description": "Screen a universe of assets by a metric across time windows (e.g. best return each quarter). Call this for ranking/comparison questions, not portfolio construction.",
    "input_schema": {
        "type": "object",
        "required": ["tickers", "start_date", "end_date", "window_freq", "metric"],
        "properties": {
            "tickers": {"type": "array", "items": {"type": "string"}, "description": "5–15 tickers to rank"},
            "start_date": {"type": "string", "description": "YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            "window_freq": {"type": "string", "enum": ["weekly", "monthly", "quarterly", "annually"]},
            "metric": {"type": "string", "enum": ["return", "sharpe", "volatility", "max_drawdown", "momentum_3m"]},
            "top_n": {"type": "integer", "description": "Top N assets per window (default 3)"},
            "screen_description": {"type": "string"},
        },
    },
}

ALL_TOOLS = [GET_ASSET_STATISTICS_TOOL, PORTFOLIO_TOOL, RUN_SCREEN_TOOL]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_dates_portfolio() -> tuple[str, str]:
    end = date.today()
    start = date(end.year - 10, end.month, end.day)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _default_dates_screener() -> tuple[str, str]:
    end = date.today()
    start = date(end.year - 3, end.month, end.day)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _extract_text(content: list) -> str:
    return " ".join(b.text for b in content if hasattr(b, "text") and b.text).strip()


def _parse_portfolio(inp: dict) -> tuple[PortfolioInput, str, dict]:
    start, end = _default_dates_portfolio()
    portfolio = PortfolioInput(
        assets=[
            AssetInput(ticker=a["ticker"], weight=a["weight"], rationale=a.get("rationale"))
            for a in inp["assets"]
        ],
        start_date=inp.get("start_date", start),
        end_date=inp.get("end_date", end),
        rebalance_frequency=inp.get("rebalance_frequency", "quarterly"),
        benchmark=inp.get("benchmark", "SPY"),
        strategy=inp.get("strategy", "custom"),
    )
    return portfolio, inp.get("strategy_summary", ""), inp.get("display_config", {})


def _parse_screen(inp: dict) -> ScreenRequest:
    start, end = _default_dates_screener()
    return ScreenRequest(
        tickers=inp["tickers"],
        start_date=inp.get("start_date", start),
        end_date=inp.get("end_date", end),
        window_freq=inp.get("window_freq", "quarterly"),
        metric=inp.get("metric", "return"),
        top_n=int(inp.get("top_n", 3)),
        screen_description=inp.get("screen_description", ""),
    )


# ── Main unified agentic loop ─────────────────────────────────────────────────

def call_unified_ai(message: str, conversation_history: list) -> dict:
    """
    Single entry point for both portfolio construction and asset screening.

    Routing is handled by Claude based on intent signals in the system prompt.
    Returns a dict with 'type' = 'portfolio' | 'screener' | 'clarification',
    plus the appropriate result fields.
    """
    from app.services.backtest_engine import run_backtest
    from app.services.metrics import calculate_metrics, build_monthly_returns_grid

    messages = [{"role": m.role, "content": m.content} for m in conversation_history]
    messages.append({"role": "user", "content": message})

    system = [{"type": "text", "text": UNIFIED_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

    for iteration in range(MAX_RESEARCH_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system,
            tools=ALL_TOOLS,
            tool_choice={"type": "auto"},
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})
        tool_calls = [b for b in response.content if b.type == "tool_use"]

        # Pure text reply (clarification question or edge case)
        if not tool_calls:
            return {
                "type": "clarification",
                "ai_response": _extract_text(response.content),
            }

        tool_results = []
        portfolio_called = False
        screen_called = False
        portfolio_input = None
        strategy_summary = ""
        display_config = {}
        screen_tc = None

        for tc in tool_calls:
            if tc.name == "get_asset_statistics":
                stats = get_asset_statistics_for_ai(
                    tickers=tc.input["tickers"],
                    start_date=tc.input["start_date"],
                    end_date=tc.input["end_date"],
                    benchmark=tc.input.get("benchmark", "SPY"),
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": json.dumps(stats, default=str),
                })

            elif tc.name == "construct_portfolio":
                portfolio_input, strategy_summary, display_config = _parse_portfolio(tc.input)
                portfolio_called = True
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": "Portfolio finalised. Backtest is running.",
                })

            elif tc.name == "run_screen":
                screen_tc = tc
                screen_called = True
                # Run the real screener immediately so we can pass results back
                screen_req = _parse_screen(tc.input)
                screen_result = run_screener(screen_req)
                result_summary = _summarise_results(screen_result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": f"Screen complete. Here are the real results:\n\n{result_summary}",
                })

        messages.append({"role": "user", "content": tool_results})

        # ── Portfolio path ────────────────────────────────────────────────────
        if portfolio_called and portfolio_input is not None:
            # Run backtest
            backtest_result = run_backtest(portfolio_input)

            # Get Claude's chat reply
            final = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=system,
                messages=messages,
            )
            ai_text = _extract_text(final.content)
            return {
                "type": "portfolio",
                "portfolio": {
                    "assets": [{"ticker": a.ticker, "weight": a.weight, "rationale": a.rationale}
                               for a in portfolio_input.assets],
                    "strategy": portfolio_input.strategy,
                    "rebalance_frequency": portfolio_input.rebalance_frequency,
                    "benchmark": portfolio_input.benchmark,
                    "strategy_summary": strategy_summary,
                    "start_date": portfolio_input.start_date,
                    "end_date": portfolio_input.end_date,
                },
                "backtest": backtest_result.model_dump(),
                "display_config": display_config,
                "ai_response": ai_text,
            }

        # ── Screener path ─────────────────────────────────────────────────────
        if screen_called:
            # Get Claude's narrative reply (it now has the real results in context)
            final = client.messages.create(
                model=MODEL,
                max_tokens=768,
                system=system,
                messages=messages,
            )
            ai_text = _extract_text(final.content)
            return {
                "type": "screener",
                "screen_result": screen_result.model_dump(),
                "ai_response": ai_text,
            }

    # Fallback
    raise ValueError("Could not determine request type. Please rephrase your question.")

