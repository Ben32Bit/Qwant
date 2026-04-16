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

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=4)

MODEL = "claude-sonnet-4-20250514"

# Optimised: most requests need at most 1 research call + 1 construct call.
# Cap at 4 to prevent runaway loops while still allowing complex data-driven strategies.
MAX_ITERATIONS = 4

import os
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))

# ── System prompt ─────────────────────────────────────────────────────────────

UNIFIED_SYSTEM_PROMPT = """You are Qwant — an AI investment analyst with two capabilities.

## ROUTING

**→ SCREEN (`run_screen`):** User wants period-by-period ranking — who led each quarter/month/year.
Triggers: "top", "best", "rank", "screen", "which performed", "each quarter/month/year", "rolling windows"

**→ PORTFOLIO (`construct_portfolio`, optionally after `get_asset_statistics`):** User wants weights, strategy, equity curve, backtest.
Triggers: "portfolio", "invest", "allocate", "backtest", weights/percentages, strategy names, "vs SPY"

Default to portfolio when ambiguous.

---

## Research rule (portfolio only)
**Skip `get_asset_statistics`** when: user gives explicit weights, names a well-known strategy ("60/40", "all-weather"), or lists specific tickers with no selection criterion.
**Call it first** only when you need data to decide weights: "most uncorrelated", "best Sharpe", "lowest vol" selection. Limit: **one call**.

---

## Tool: run_screen
Call once. All parameters from the request.

**metric:** return | sharpe | volatility | max_drawdown | momentum_3m
**window_freq (default quarterly):** weekly | monthly | quarterly | annually
**top_n (default 3):** 1 | 3 | 5
**end_date:** always forced to today. Default start: 3 years ago.

**Pre-cached universes (prefer when not specified):**
- Sectors: XLK XLF XLE XLV XLY XLP XLU XLB XLI XLRE XLC
- Tech/FAANG: META AAPL AMZN NFLX GOOGL MSFT NVDA AVGO
- International: VTI EFA EEM VEA VWO
- Asset classes: SPY TLT GLD VNQ DBC HYG EEM
- Bonds: TLT IEF SHY LQD HYG AGG BND | Commodities: GLD SLV USO DBC
- Large-cap: AAPL MSFT NVDA AMZN META GOOGL TSLA AVGO NFLX AMD ORCL CRM
- Financials: JPM BAC GS V MA

---

## Tool: construct_portfolio
- Real US-listed tickers only. Use VIXY/VXX, not ^VIX.
- Negative weights = short. Weights may sum >1 (leverage).
- Default: 10-year range ending today, quarterly rebalance, SPY benchmark.
- Always include `display_config` with sections and narrative.

**Long/Short & Market-Neutral:** When user implies a short ("long/short", "pair trade", "market neutral", "hedge", "short X"):
- Use negative weights. Never use a reduced positive weight instead.
- Dollar-neutral pair: +1.0 long / −1.0 short. Use daily rebalance_frequency.
- 130/30: longs ~1.3, shorts ~−0.3. Market-neutral: net weight ≈ 0.

**display_config.sections — always:** equity_curve, drawdown, metrics_summary, ff5_decomposition
Add: weight_drift (≥2 assets) | correlation_matrix (≥3 assets, diversification focus) | rolling_metrics (risk consistency) | monthly_heatmap (multi-year) | full_metrics (benchmark focus)

**display_config.narrative — full markdown including:** What Was Built (tickers/weights/strategy/dates), Key Findings (CAGR, Sharpe, drawdown vs benchmark, DSR if relevant), Risks & Caveats (overfitting, regime risk, concentration), Suggested Next Steps (1–2 refinements).

---

## Backtest Integrity
DSR > 0.95 = strong alpha. DSR 0.90–0.95 = moderate. DSR < 0.90 = likely data-mined.
In-sample optimisation (max Sharpe, min variance) inflates Sharpe. Short backtests (<3y) risk spurious results.

---

## Output Format

**After run_screen:** 4–5 markdown bullets (`-` prefix, one per line) referencing actual tickers and window counts. End: "→ Click **Backtest Rotation Strategy** to test this as a live momentum portfolio"

**After construct_portfolio:** 3–5 markdown bullets (`-` prefix, one per line). Cover: what was built + headline result, one statistical insight, one next step. Full analysis in display_config.narrative — do not repeat it here."""


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
                                     "monthly_heatmap", "correlation_matrix", "rolling_metrics", "weight_drift",
                                     "ff5_decomposition"],
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
    start, _ = _default_dates_portfolio()
    # end_date is always today — never trust what the AI or user provides for this
    today = date.today().strftime("%Y-%m-%d")
    portfolio = PortfolioInput(
        assets=[
            AssetInput(ticker=a["ticker"], weight=a["weight"], rationale=a.get("rationale"))
            for a in inp["assets"]
        ],
        start_date=inp.get("start_date", start),
        end_date=today,          # hard requirement: always ends today
        rebalance_frequency=inp.get("rebalance_frequency", "quarterly"),
        benchmark=inp.get("benchmark", "SPY"),
        strategy=inp.get("strategy", "custom"),
    )
    return portfolio, inp.get("strategy_summary", ""), inp.get("display_config", {})


def _parse_screen(inp: dict) -> ScreenRequest:
    start, _ = _default_dates_screener()
    # end_date is always today — never trust what the AI or user provides for this
    today = date.today().strftime("%Y-%m-%d")
    return ScreenRequest(
        tickers=inp["tickers"],
        start_date=inp.get("start_date", start),
        end_date=today,          # hard requirement: always ends today
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
