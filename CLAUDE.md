# Portfolio Backtester — AI-Powered Portfolio Construction & Backtesting Platform

## Instructions for Claude Code

**READ THIS FILE FIRST** before making any changes to the codebase.

After every meaningful change (new feature, bug fix, refactor, config change), update `context-log.md` in the project root with:
- Date & time of change
- What was changed and why
- Files affected
- Any decisions made or trade-offs chosen
- Current state of the project (what works, what's pending)

If `context-log.md` does not exist, create it with an initial entry summarizing the project setup.

Always keep this reference file (`PROJECT-REFERENCE.md`) and `context-log.md` in sync. If a design decision in this file is overridden during development, note the override in both files.

---

## 1. Product Vision

A **free, no-signup** web application where users describe portfolios in natural language and get instant backtesting results with interactive charts. The AI interprets user intent, resolves it to concrete tickers, weights (including negative/short and leveraged positions), and runs historical backtests — all in one conversational flow.

### Core Principles
- **Speed first** — sub-second UI interactions, backtest results in <2s
- **No barriers** — no signup, no login, no paywall
- **Conversational** — users describe what they want, AI builds it
- **Professional-grade output** — metrics and charts that a portfolio manager would recognize

---

## 2. Key Features

### 2.1 AI-Powered Portfolio Construction
- Natural language input: "60/40 US stocks and bonds since 2015"
- AI resolves intent → concrete tickers + weights + date range
- Conversational refinement: "now add 10% gold" / "make it more aggressive"
- Full conversation context preserved across turns

### 2.2 Portfolio Weights — Shorting & Leverage
- Weights can be **negative** (short positions): e.g., AAPL: -0.20 means 20% short
- Weights can sum to **>1.0** (leverage): e.g., total weight 1.5 = 150% exposure
- Weights can sum to **<1.0** (partial investment, rest in cash)
- UI must clearly indicate short positions (red/negative) and leverage warnings
- Backtest engine must correctly compute returns for short positions: short_return = -1 × long_return

### 2.3 Portfolio Rebalancing
- Supported frequencies: daily, weekly, monthly, quarterly, annually, none (buy-and-hold drift)
- Rebalancing logic: on each rebalance date, reset portfolio weights to target allocation
- Between rebalance dates, weights drift based on relative asset performance
- Track turnover from rebalancing (sum of absolute weight changes)
- Option for threshold-based rebalancing: rebalance only when any weight drifts >X% from target

### 2.4 Strategy Construction
- **Minimum Variance Portfolio**: optimize weights to minimize portfolio variance using covariance matrix
- **Maximum Sharpe Portfolio**: optimize weights to maximize Sharpe ratio
- **Risk Parity**: weight assets so each contributes equal risk
- **Inverse Volatility**: weight inversely proportional to each asset's historical volatility
- **Equal Weight**: simple 1/N allocation
- All strategies use rolling or full-period covariance matrix estimation
- Covariance matrix computation: use exponentially weighted or simple rolling window

### 2.5 Benchmark Overlays
- Compare against major benchmark ETFs: SPY, QQQ, IWM, AGG, EFA, EEM, GLD, TLT, VTI, BND
- Compare against other user-defined portfolios (overlay multiple backtests)
- Toggle benchmarks on/off on the equity curve chart
- Relative performance chart (portfolio / benchmark indexed to 100)
- Active return and tracking error vs selected benchmark

### 2.6 Portfolio Metrics
All metrics calculated and displayed in a dashboard:

**Return Metrics:**
- Total Return (%)
- CAGR (Compound Annual Growth Rate)
- Best / Worst Year
- Best / Worst Month
- Monthly return heatmap (year × month grid)

**Risk Metrics:**
- Annualized Volatility
- Max Drawdown (%) and duration (days)
- Downside Deviation
- Value at Risk (95% and 99%)
- Conditional VaR (Expected Shortfall)

**Risk-Adjusted Metrics:**
- Sharpe Ratio (vs risk-free rate)
- Sortino Ratio
- Calmar Ratio (CAGR / Max Drawdown)
- Information Ratio (vs benchmark)
- Treynor Ratio

**Benchmark-Relative Metrics:**
- Beta (vs selected benchmark)
- Alpha (Jensen's Alpha)
- R-squared
- Tracking Error
- Active Return
- Up Capture / Down Capture Ratio

### 2.7 Visualization / Charts
- **Equity Curve**: log and linear scale toggle, multiple overlays
- **Drawdown Chart**: underwater chart showing drawdown periods
- **Rolling Metrics**: rolling Sharpe, rolling volatility, rolling beta (configurable window)
- **Monthly Returns Heatmap**: color-coded grid of monthly returns
- **Asset Correlation Matrix**: heatmap of pairwise correlations
- **Weight Drift Chart**: how portfolio weights change between rebalances
- **Distribution Chart**: histogram of daily/monthly returns with normal overlay

---

## 3. Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────┐
│           React Frontend            │
│  (Vite + Tailwind + Recharts)       │
│                                     │
│  ┌───────────┐  ┌────────────────┐  │
│  │ Chat Panel │  │ Results Panel  │  │
│  │ (left 40%) │  │ (right 60%)   │  │
│  └─────┬─────┘  └───────▲────────┘  │
│        │                │           │
└────────┼────────────────┼───────────┘
         │  POST /api/*   │ JSON response
         ▼                │
┌────────────────────────────────────┐
│        FastAPI Backend             │
│                                    │
│  ┌──────────┐  ┌────────────────┐  │
│  │ AI Layer │  │ Backtest Engine│  │
│  │(Anthropic│  │ (Pandas/NumPy) │  │
│  │  API)    │  │                │  │
│  └────┬─────┘  └───────▲────────┘  │
│       │                │           │
│  ┌────▼────────────────┴────────┐  │
│  │     Data Layer (yfinance)    │  │
│  │     + LRU Cache              │  │
│  └──────────────────────────────┘  │
└────────────────────────────────────┘
```

### 3.2 Frontend (React + Vite)

**Directory structure:**
```
frontend/
├── src/
│   ├── components/
│   │   ├── Chat/
│   │   │   ├── ChatPanel.jsx         # Main chat container
│   │   │   ├── MessageBubble.jsx     # Individual message
│   │   │   ├── PromptSuggestions.jsx  # Example prompt chips
│   │   │   └── PortfolioCard.jsx     # Inline portfolio display in chat
│   │   ├── Dashboard/
│   │   │   ├── ResultsPanel.jsx      # Main results container
│   │   │   ├── EquityCurve.jsx       # Main equity chart
│   │   │   ├── DrawdownChart.jsx     # Underwater drawdown
│   │   │   ├── MetricsCards.jsx      # Key metrics grid
│   │   │   ├── MonthlyHeatmap.jsx    # Monthly returns heatmap
│   │   │   ├── CorrelationMatrix.jsx # Asset correlation heatmap
│   │   │   ├── RollingMetrics.jsx    # Rolling Sharpe/vol/beta
│   │   │   ├── WeightDrift.jsx       # Weight evolution chart
│   │   │   └── BenchmarkSelector.jsx # Toggle benchmark overlays
│   │   ├── Portfolio/
│   │   │   ├── PortfolioTable.jsx    # Editable weights table
│   │   │   └── WeightBar.jsx        # Visual weight bar (green=long, red=short)
│   │   └── Layout/
│   │       ├── App.jsx
│   │       ├── SplitView.jsx         # Two-panel layout
│   │       └── ThemeToggle.jsx
│   ├── hooks/
│   │   ├── useBacktest.js            # API call + state management
│   │   └── useChat.js                # Conversation history management
│   ├── utils/
│   │   ├── formatters.js             # Number/date/percent formatting
│   │   └── chartConfig.js            # Shared Recharts config
│   ├── styles/
│   │   └── globals.css
│   └── main.jsx
├── index.html
├── vite.config.js
├── tailwind.config.js
└── package.json
```

**Chart library:** Recharts (primary) with Plotly as fallback for correlation heatmaps. Both support interactive tooltips, zoom, and responsive sizing.

**Design direction:** Dark-theme financial terminal aesthetic. Monospace numbers, high contrast charts, minimal chrome. Think Bloomberg terminal meets modern web. Use a distinctive display font for headers, monospace for data.

**Performance requirements:**
- Skeleton loaders during API calls
- Optimistic UI updates where possible
- Virtualize long chat histories
- Debounce input (300ms)
- Memoize chart components to prevent unnecessary re-renders

### 3.3 Backend (FastAPI + Python)

**Directory structure:**
```
backend/
├── app/
│   ├── main.py                  # FastAPI app, CORS, routes
│   ├── routers/
│   │   ├── chat.py              # POST /api/chat — main AI endpoint
│   │   ├── backtest.py          # POST /api/backtest — direct backtest
│   │   └── data.py              # GET /api/tickers/search — ticker search
│   ├── services/
│   │   ├── ai_service.py        # Anthropic API calls with tool use
│   │   ├── backtest_engine.py   # Core backtest computation
│   │   ├── metrics.py           # All portfolio metric calculations
│   │   ├── optimization.py      # Min variance, max Sharpe, risk parity
│   │   └── data_service.py      # yfinance wrapper + caching
│   ├── models/
│   │   ├── portfolio.py         # Pydantic models for portfolio
│   │   ├── backtest_result.py   # Pydantic models for results
│   │   └── chat.py              # Pydantic models for chat messages
│   └── utils/
│       ├── cache.py             # LRU cache for price data
│       └── constants.py         # Benchmark tickers, risk-free rate
├── requirements.txt
├── Dockerfile
└── .env.example
```

**Key dependencies:**
```
fastapi
uvicorn
anthropic
yfinance
pandas
numpy
scipy              # for optimization (minimize)
python-dotenv
```

### 3.4 AI Layer — Anthropic Tool Use

**Model:** `claude-sonnet-4-20250514` (fast, cheap, excellent at tool use)

**System prompt for the AI service:**
```
You are a portfolio construction assistant. The user will describe a portfolio
or investment strategy in natural language. Your job is to interpret their
intent and call the construct_portfolio tool with concrete, investable
parameters.

Rules:
- Use real, currently tradable ticker symbols (US-listed ETFs and stocks)
- Weights can be negative (short positions) and can sum to >1.0 (leverage)
- Default date range: last 10 years to today if not specified
- Default rebalancing: quarterly if not specified
- Default benchmark: SPY if not specified
- If the user asks for a strategy (min variance, max sharpe, risk parity),
  set the "strategy" field and provide candidate tickers. The backend will
  compute optimal weights.
- Always include a brief rationale for each holding
- If the request is ambiguous, make reasonable assumptions and state them
```

**Tool definition:**
```python
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
```

### 3.5 Backtest Engine — Core Math

**Located in:** `backend/app/services/backtest_engine.py`

```python
# Pseudocode for the core backtest loop

def run_backtest(prices_df, weights, rebalance_freq, initial_capital=10000):
    """
    prices_df: DataFrame of adjusted close prices, columns = tickers, index = dates
    weights: dict of {ticker: weight} — can be negative (short), can sum to != 1
    rebalance_freq: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'annually' | 'none'
    """

    daily_returns = prices_df.pct_change().dropna()
    rebalance_dates = get_rebalance_dates(daily_returns.index, rebalance_freq)

    # Initialize
    current_weights = pd.Series(weights)
    portfolio_values = [initial_capital]

    for date in daily_returns.index:
        # If rebalance date, reset weights to target
        if date in rebalance_dates:
            current_weights = pd.Series(weights)

        # Daily portfolio return = sum(weight_i * return_i)
        daily_ret = (current_weights * daily_returns.loc[date]).sum()
        new_value = portfolio_values[-1] * (1 + daily_ret)
        portfolio_values.append(new_value)

        # Drift weights based on individual asset returns
        current_weights = current_weights * (1 + daily_returns.loc[date])
        current_weights = current_weights / current_weights.sum()  # re-normalize

    return pd.Series(portfolio_values[1:], index=daily_returns.index)
```

**Note on short positions:** When weight is negative, the return contribution is automatically correct: negative_weight × positive_return = negative contribution (short loses when asset goes up). No special handling needed beyond allowing negative weights.

**Note on leverage:** When weights sum to >1.0, the portfolio is implicitly leveraged. Returns are amplified proportionally. Consider adding a borrowing cost parameter (e.g., Fed Funds Rate) that reduces returns proportional to leverage amount.

### 3.6 Optimization Engine

**Located in:** `backend/app/services/optimization.py`

Uses `scipy.optimize.minimize` for portfolio optimization.

```python
from scipy.optimize import minimize

def min_variance_portfolio(returns_df, allow_short=True):
    """Find weights that minimize portfolio variance."""
    n = len(returns_df.columns)
    cov_matrix = returns_df.cov() * 252  # annualize

    def portfolio_variance(weights):
        return weights @ cov_matrix @ weights

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(-1, 1) if allow_short else (0, 1) for _ in range(n)]
    initial = np.array([1/n] * n)

    result = minimize(portfolio_variance, initial, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    return dict(zip(returns_df.columns, result.x))
```

Similar implementations for max_sharpe (maximize return/std), risk_parity (equalize marginal risk contribution), and inverse_vol.

### 3.7 Metrics Calculations

**Located in:** `backend/app/services/metrics.py`

All metrics take a daily returns series and optionally a benchmark returns series.

Key formulas:
- **CAGR** = (final_value / initial_value) ^ (252 / trading_days) - 1
- **Volatility** = std(daily_returns) × √252
- **Sharpe** = (CAGR - risk_free_rate) / volatility
- **Sortino** = (CAGR - risk_free_rate) / downside_deviation
- **Max Drawdown** = max(1 - portfolio_value / cumulative_max)
- **Calmar** = CAGR / |max_drawdown|
- **Beta** = cov(portfolio, benchmark) / var(benchmark)
- **Alpha** = CAGR - (risk_free + beta × (benchmark_CAGR - risk_free))
- **R²** = correlation(portfolio, benchmark)²
- **Information Ratio** = mean(active_return) / std(active_return) × √252
- **Tracking Error** = std(portfolio_return - benchmark_return) × √252
- **VaR(95%)** = percentile(daily_returns, 5) × √252
- **CVaR(95%)** = mean(daily_returns[daily_returns <= VaR_95]) × √252
- **Up Capture** = CAGR(portfolio on up benchmark days) / CAGR(benchmark on up days)
- **Down Capture** = CAGR(portfolio on down benchmark days) / CAGR(benchmark on down days)
- **Treynor** = (CAGR - risk_free) / beta

---

## 4. API Endpoints

### POST /api/chat
Main conversational endpoint. Accepts natural language, returns portfolio + backtest results.

**Request:**
```json
{
  "message": "60/40 US stocks and bonds over the last 10 years",
  "conversation_history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "current_portfolio": null
}
```

**Response:**
```json
{
  "portfolio": {
    "assets": [
      {"ticker": "VTI", "weight": 0.60, "rationale": "Total US stock market"},
      {"ticker": "BND", "weight": 0.40, "rationale": "US investment-grade bonds"}
    ],
    "strategy": "custom",
    "rebalance_frequency": "quarterly",
    "benchmark": "SPY"
  },
  "backtest": {
    "equity_curve": [{"date": "2015-01-02", "value": 10000}, ...],
    "benchmark_curve": [{"date": "2015-01-02", "value": 10000}, ...],
    "drawdown_series": [{"date": "2015-01-02", "drawdown": 0.0}, ...],
    "weight_history": [...],
    "monthly_returns": {"2015": {"Jan": 0.02, "Feb": -0.01, ...}, ...},
    "metrics": {
      "total_return": 0.95,
      "cagr": 0.072,
      "volatility": 0.098,
      "sharpe": 0.61,
      "sortino": 0.85,
      "max_drawdown": -0.18,
      "max_drawdown_duration_days": 45,
      "calmar": 0.40,
      "beta": 0.62,
      "alpha": 0.015,
      "r_squared": 0.89,
      "information_ratio": 0.25,
      "tracking_error": 0.065,
      "var_95": -0.015,
      "cvar_95": -0.022,
      "up_capture": 0.72,
      "down_capture": 0.55,
      "treynor": 0.097,
      "best_year": {"year": 2021, "return": 0.18},
      "worst_year": {"year": 2022, "return": -0.15},
      "best_month": {"month": "2020-04", "return": 0.08},
      "worst_month": {"month": "2020-03", "return": -0.09}
    }
  },
  "ai_response": "I've built a classic 60/40 portfolio using VTI for broad US equity exposure and BND for investment-grade bonds...",
  "strategy_summary": "Classic 60/40 balanced portfolio"
}
```

### POST /api/backtest
Direct backtest endpoint (bypasses AI). Used for manual weight adjustments in the UI.

**Request:**
```json
{
  "assets": [
    {"ticker": "AAPL", "weight": 0.5},
    {"ticker": "MSFT", "weight": 0.3},
    {"ticker": "SPY", "weight": -0.2}
  ],
  "start_date": "2020-01-01",
  "end_date": "2025-01-01",
  "rebalance_frequency": "monthly",
  "benchmark": "SPY",
  "initial_capital": 10000,
  "strategy": "custom"
}
```

### GET /api/tickers/search?q=apple
Returns matching tickers for autocomplete in manual mode.

### GET /api/benchmarks
Returns list of available benchmark ETFs with metadata.

---

## 5. Cost Management

- **Model:** Always use `claude-sonnet-4-20250514` — fast, cheap ($3/$15 per M tokens)
- **Cache common queries:** Hash normalized prompts, cache AI response + backtest results
- **Cache price data:** LRU cache with TTL of 1 hour for intraday, 24 hours for historical
- **Rate limit:** 15 requests/hour per IP (no signup, so IP-based)
- **Estimated cost:** ~$0.005 per request → $25/day at 5000 requests

---

## 6. Deployment

| Layer | Service | Plan |
|-------|---------|------|
| Frontend | Vercel or Netlify | Free |
| Backend | Railway or Render | Free tier (may need starter for always-on) |
| Domain | Optional | ~$10/year |

**Environment variables (backend):**
```
ANTHROPIC_API_KEY=sk-ant-...
RISK_FREE_RATE=0.05
CACHE_TTL_HOURS=24
RATE_LIMIT_PER_HOUR=15
CORS_ORIGINS=https://your-frontend-domain.com
```

---

## 7. Development Phases

### Phase 1: Foundation
- [ ] Project scaffolding (Vite + FastAPI)
- [ ] Basic backtest engine (long-only, no rebalancing)
- [ ] Single equity curve chart
- [ ] Direct POST /api/backtest endpoint
- [ ] Basic metrics (CAGR, vol, Sharpe, max DD)

### Phase 2: AI Integration
- [ ] Anthropic API integration with tool use
- [ ] POST /api/chat endpoint
- [ ] Chat UI panel
- [ ] Conversation history + refinement flow

### Phase 3: Advanced Features
- [ ] Short positions + leverage support
- [ ] Rebalancing logic (all frequencies)
- [ ] Benchmark overlays (multi-line chart)
- [ ] Full metrics dashboard (all metrics from section 2.6)
- [ ] Portfolio optimization strategies (min var, max Sharpe, etc.)

### Phase 4: Visualization Polish
- [ ] Monthly returns heatmap
- [ ] Correlation matrix
- [ ] Rolling metrics charts
- [ ] Weight drift chart
- [ ] Return distribution histogram
- [ ] Dark/light theme toggle

### Phase 5: Production Hardening
- [ ] Rate limiting
- [ ] Caching layer
- [ ] Error handling + input validation
- [ ] Loading states + skeleton UI
- [ ] Responsive / mobile layout
- [ ] Deployment

---

## 8. Design Reference

**Aesthetic:** Dark financial terminal. Think Bloomberg Terminal meets modern web.

**Color palette (CSS variables):**
```css
--bg-primary: #0a0a0f;
--bg-secondary: #12121a;
--bg-card: #1a1a25;
--text-primary: #e0e0e8;
--text-secondary: #8888a0;
--accent-green: #00d4aa;       /* positive returns, long positions */
--accent-red: #ff4757;         /* negative returns, short positions */
--accent-blue: #4a9eff;        /* primary accent, links, buttons */
--accent-yellow: #ffd43b;      /* warnings, leverage indicators */
--chart-line-1: #4a9eff;       /* portfolio */
--chart-line-2: #8888a0;       /* benchmark */
--chart-line-3: #00d4aa;       /* overlay 1 */
--chart-line-4: #ffd43b;       /* overlay 2 */
--border: #2a2a3a;
```

**Typography:**
- Headers: JetBrains Mono or IBM Plex Mono (bold, distinctive)
- Body: IBM Plex Sans or equivalent clean sans-serif
- Numbers/data: JetBrains Mono (tabular figures)

**Key UI patterns:**
- Monospace numbers everywhere for data alignment
- Color-code all returns: green=positive, red=negative
- Short positions shown with a red badge and negative sign
- Leverage indicator when total weight > 100%
- Skeleton loaders that match the shape of the content they replace
- Subtle animations on metric card updates (count-up effect)

---

## 9. Testing Notes

- Validate that shorting works: a 100% short SPY portfolio should mirror SPY returns inverted
- Validate leverage: a 200% SPY portfolio should have ~2x SPY volatility
- Validate rebalancing: compare quarterly vs no-rebalance on a diverging pair
- Edge cases: single asset portfolio, 100% cash (empty portfolio), all-short portfolio
- Verify metrics against a known source (e.g., Portfolio Visualizer) for a simple case
