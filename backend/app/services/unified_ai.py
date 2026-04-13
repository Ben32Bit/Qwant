import anthropic
import os
import json
from datetime import date
from app.models.portfolio import PortfolioInput, AssetInput
from app.models.screener import ScreenRequest
from app.services.data_service import get_asset_statistics_for_ai
from app.services.screener_engine import run_screener
from app.services.screener_ai import _summarise_results
from app.services.backtest_engine import run_full_backtest
from app.services.optimization import apply_strategy
from app.services.data_service import fetch_prices

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-sonnet-4-20250514"

# Optimised: most requests need at most 1 research call + 1 construct call.
# Cap at 4 to prevent runaway loops while still allowing complex data-driven strategies.
MAX_ITERATIONS = 4

import os
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))

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

### → BUILD PORTFOLIO  (call `construct_portfolio`, optionally after `get_asset_statistics`)
User wants to **build, allocate, and backtest a portfolio** — weights, strategy, equity curve.

**Signal phrases:** "portfolio", "invest", "allocate", "build", "backtest", weights or percentages, strategy names (60/40, risk parity, all-weather, FAANG, max Sharpe, min variance), "vs SPY", "vs benchmark"

**Examples:**
- "60/40 US stocks and bonds over 10 years"
- "Max Sharpe from SPY, TLT, GLD, EEM"
- "Equal weight FAANG vs SPY since 2018"

### When ambiguous
Default to portfolio construction unless the user clearly asks for a window-by-window ranking.

---

## Research rule (portfolio path only)
**Skip `get_asset_statistics` and go straight to `construct_portfolio` when:**
- The user gives explicit weights ("60% SPY, 40% BND")
- The portfolio is a well-known strategy ("60/40", "all-weather", "FAANG equal weight")
- The user names specific tickers with no selection criterion

**Call `get_asset_statistics` first only when you genuinely need data to decide weights:**
- "most uncorrelated", "best Sharpe", "lowest volatility" selection
- Momentum or factor-based picking from a large candidate set
- When comparing multiple strategies before committing

Limit to **one** `get_asset_statistics` call. If the data is sufficient after one call, proceed immediately to `construct_portfolio`.

---

## Tool: run_screen
Call this ONCE. Pick all parameters directly from the request.

### Metric Guide
- "top returns / best performance" → metric="return"
- "best risk-adjusted / best Sharpe" → metric="sharpe"
- "least volatile / most stable" → metric="volatility"
- "smallest drawdown / most resilient" → metric="max_drawdown"
- "momentum / recent trend" → metric="momentum_3m"

### Universe Guide
- "sectors / sector ETFs" → XLK, XLF, XLE, XLV, XLY, XLP, XLU, XLB, XLI, XLRE, XLC
- "mega-cap tech / FAANG" → META, AAPL, AMZN, NFLX, GOOGL (add MSFT, NVDA for broader tech)
- "global / international" → VTI, EFA, EEM, VEA, VWO, EWJ, EWZ, FXI
- "asset classes" → SPY, TLT, GLD, VNQ, DBC, HYG, EEM
- "bonds" → TLT, IEF, SHY, LQD, HYG, AGG, BND
- "commodities" → GLD, SLV, USO, DBC, CPER
- Explicit tickers → use exactly those

### Window Frequency — default "quarterly"
- "quarterly / each quarter" → "quarterly"
- "monthly / each month" → "monthly"
- "annually / each year" → "annually"
- "weekly" → "weekly"

### top_n — default 3
- "top 1 / best / winner" → 1  |  "top 3" → 3  |  "top 5" → 5

### Date Range — default last 3 years

---

## Tool: get_asset_statistics
Research tool. Use sparingly — only when you need real data to decide weights.
Call at most once before `construct_portfolio`.

---

## Tool: construct_portfolio
Finalise a portfolio. Rules:
- Real, US-listed tickers only. Never use ^VIX — use VIXY or VXX.
- Weights can be negative (short) and sum >1.0 (leverage)
- Default date range: last 10 years | Default rebalance: quarterly | Default benchmark: SPY
- Set `display_config` with appropriate sections and the full markdown narrative

### display_config.sections
Always include: equity_curve, drawdown, metrics_summary. Add:
- weight_drift when ≥2 assets
- correlation_matrix when ≥3 assets and diversification is the goal
- rolling_metrics when risk consistency over time matters
- monthly_heatmap for multi-year strategies
- full_metrics when benchmark comparison is the focus

### display_config.narrative (full markdown, shown in results panel)
```
## What Was Built
- [tickers, weights, strategy, rebalance, date range]

## Key Findings
- [CAGR and Sharpe — be specific with numbers from the backtest]
- [Drawdown, volatility vs benchmark]
- [DSR commentary if relevant]

## Risks & Caveats
- [Bailey/overfitting warnings if applicable]
- [Regime sensitivity, concentration risk, etc.]

## Suggested Next Steps
- [1–2 actionable refinements the user could try]
```

---

## Backtest Integrity (Bailey & Lopez de Prado)
- DSR > 0.95 → stronger evidence of genuine alpha
- DSR 0.90–0.95 → moderate confidence
- DSR < 0.90 → likely data-mining artefact
- In-sample optimisation (max Sharpe, min variance) inflates reported Sharpe — real out-of-sample will be lower
- Short backtests (<3 years) have high chance of spurious Sharpe

---

## Output Format

### After `run_screen` + real results → write 4–5 bullets in chat
Reference actual tickers and window counts. End with:
"→ Click **Backtest Rotation Strategy** to test this as a live momentum portfolio"

### After `construct_portfolio` → write 3–5 bullets in chat (no headers)
- What was built + headline result
- One statistical insight (Sharpe, DSR, or benchmark comparison)
- 1 suggested next step
Full analysis goes in display_config.narrative — do NOT repeat it in the chat bullets."""


# ── Tool definitions ──────────────────────────────────────────────────────────

GET_ASSET_STATISTICS_TOOL = {
    "name": "get_asset_statistics",
    "description": "Fetch real historical statistics (return, volatility, Sharpe, drawdown, correlations) for a list of tickers. Use ONLY when you need data to decide weights — skip for explicit portfolios.",
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
    "description": "Finalise a portfolio for backtesting. Call after any needed research, or directly for explicit portfolios.",
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
                    "narrative": {"type": "string", "description": "Full markdown analysis for the results panel"},
                },
            },
        },
    },
}

RUN_SCREEN_TOOL = {
    "name": "run_screen",
    "description": "Screen a universe of assets by a metric across time windows. Use for ranking/comparison questions, not portfolio construction.",
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
    Claude routes by calling the appropriate tool.

    Portfolio path:  [optional get_asset_statistics] → construct_portfolio
                     → run_full_backtest → chat reply
    Screener path:   run_screen → real screener results fed back → chat reply

    Returns dict with 'type' = 'portfolio' | 'screener' | 'clarification'.
    """
    messages = [{"role": m.role, "content": m.content} for m in conversation_history]
    messages.append({"role": "user", "content": message})

    system = [{"type": "text", "text": UNIFIED_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

    portfolio_input = None
    strategy_summary = ""
    display_config = {}
    screen_result = None

    for _ in range(MAX_ITERATIONS):
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

        # No tool call → clarifying question or text response
        if not tool_calls:
            return {
                "type": "clarification",
                "ai_response": _extract_text(response.content),
            }

        tool_results = []
        portfolio_called = False
        screen_called = False

        for tc in tool_calls:

            # ── Research tool ─────────────────────────────────────────────────
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

            # ── Portfolio finalisation ────────────────────────────────────────
            elif tc.name == "construct_portfolio":
                portfolio_input, strategy_summary, display_config = _parse_portfolio(tc.input)
                portfolio_called = True
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": "Portfolio accepted. Backtest is running.",
                })

            # ── Screener ──────────────────────────────────────────────────────
            elif tc.name == "run_screen":
                screen_called = True
                screen_req = _parse_screen(tc.input)
                screen_result = run_screener(screen_req)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": f"Screen complete. Real results:\n\n{_summarise_results(screen_result)}",
                })

        messages.append({"role": "user", "content": tool_results})

        # ── Portfolio path: run backtest then get short chat reply ────────────
        if portfolio_called and portfolio_input is not None:

            # Apply optimisation strategy (min_variance / max_sharpe / etc.)
            if portfolio_input.strategy != "custom":
                tickers = [a.ticker for a in portfolio_input.assets]
                prices = fetch_prices(
                    list(set(tickers + [portfolio_input.benchmark])),
                    portfolio_input.start_date,
                    portfolio_input.end_date,
                )
                port_prices = prices[[t for t in tickers if t in prices.columns]]
                returns_df = port_prices.pct_change().dropna()
                current_weights = {a.ticker: a.weight for a in portfolio_input.assets}
                optimised = apply_strategy(
                    portfolio_input.strategy, returns_df, current_weights, RISK_FREE_RATE
                )
                for asset in portfolio_input.assets:
                    if asset.ticker in optimised:
                        asset.weight = float(optimised[asset.ticker])

            backtest_result = run_full_backtest(portfolio_input)

            # Short chat bubble reply — narrative is already inside display_config
            final = client.messages.create(
                model=MODEL,
                max_tokens=512,   # 3-5 bullets only
                system=system,
                messages=messages,
            )
            ai_text = _extract_text(final.content)

            return {
                "type": "portfolio",
                "portfolio": {
                    "assets": [
                        {"ticker": a.ticker, "weight": a.weight, "rationale": a.rationale}
                        for a in portfolio_input.assets
                    ],
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

        # ── Screener path: get data-aware narrative then return ───────────────
        if screen_called and screen_result is not None:
            final = client.messages.create(
                model=MODEL,
                max_tokens=512,   # 4-5 bullets only
                system=system,
                messages=messages,
            )
            return {
                "type": "screener",
                "screen_result": screen_result.model_dump(),
                "ai_response": _extract_text(final.content),
            }

    raise ValueError("Could not complete the request. Please rephrase your question.")
