# Context Log — Portfolio Backtester

This file tracks all changes, decisions, and project state. Updated by Claude Code after every meaningful change.

---

## 2026-04-12 — Project Initialization

**What:** Created `PROJECT-REFERENCE.md` — the master reference document for the entire project.

**Key decisions made during ideation:**
- Architecture: React (Vite) frontend + FastAPI (Python) backend
- AI layer: Anthropic API with tool use (Claude Sonnet) for natural language portfolio construction
- Charts: Recharts (primary), Plotly for heatmaps
- Data source: yfinance with aggressive caching
- No signup required — rate limit by IP
- Supports shorting (negative weights) and leverage (weights summing >1.0)
- Portfolio optimization: min variance, max Sharpe, risk parity, inverse vol, equal weight
- Full metrics suite: beta, alpha, Sharpe, Sortino, Calmar, VaR, CVaR, capture ratios, etc.
- Benchmark overlays with major ETFs (SPY, QQQ, AGG, etc.)
- Rebalancing: daily/weekly/monthly/quarterly/annually/none + threshold-based option
- Design: dark financial terminal aesthetic

**Files created:**
- `PROJECT-REFERENCE.md` — master architecture & feature reference
- `context-log.md` — this file

**Current state:** Planning complete. No code written yet. Ready for Phase 1 implementation.

**Next steps:** Scaffold frontend (Vite + React + Tailwind) and backend (FastAPI) projects, implement basic backtest engine.

---

## 2026-04-12 — Phase 1 + 2 Implementation (Foundation + AI Integration)

**What:** Full scaffold of both backend (FastAPI/Python) and frontend (React/Vite/Tailwind).
Implemented Phases 1 and 2 simultaneously since they are tightly coupled.

**Backend files created:**
- `backend/app/main.py` — FastAPI app with CORS, includes all routers
- `backend/app/routers/chat.py` — POST /api/chat (AI + backtest pipeline)
- `backend/app/routers/backtest.py` — POST /api/backtest (direct, no AI)
- `backend/app/routers/data.py` — GET /api/tickers/search + GET /api/benchmarks
- `backend/app/services/ai_service.py` — Anthropic API with tool use + prompt caching
- `backend/app/services/backtest_engine.py` — Core backtest loop (drift weights, rebalancing)
- `backend/app/services/metrics.py` — Full metrics suite (Sharpe, Sortino, Calmar, VaR, CVaR, beta, alpha, R², capture ratios, etc.)
- `backend/app/services/optimization.py` — min_variance, max_sharpe, risk_parity, inverse_vol, equal_weight (scipy.optimize)
- `backend/app/services/data_service.py` — yfinance wrapper with TTLCache (24h)
- `backend/app/models/portfolio.py` — PortfolioInput, AssetInput Pydantic models
- `backend/app/models/backtest_result.py` — BacktestResult, PortfolioMetrics Pydantic models
- `backend/app/models/chat.py` — ChatRequest, ChatResponse Pydantic models
- `backend/app/utils/constants.py` — BENCHMARK_TICKERS, defaults
- `backend/app/utils/cache.py` — TTLCache instances for price + AI data
- `backend/requirements.txt` — All Python dependencies pinned
- `backend/.env.example` — Required environment variables

**Frontend files created:**
- `frontend/package.json` — React 18, Recharts, Axios, Tailwind
- `frontend/vite.config.js` — Vite config with /api proxy to localhost:8000
- `frontend/tailwind.config.js` — Custom color palette (dark terminal theme)
- `frontend/index.html` — Loads JetBrains Mono + IBM Plex Sans fonts
- `frontend/src/main.jsx` — React root
- `frontend/src/styles/globals.css` — CSS variables, skeleton animation, color classes
- `frontend/src/hooks/useChat.js` — Chat state + API calls
- `frontend/src/hooks/useBacktest.js` — Direct backtest API calls
- `frontend/src/utils/formatters.js` — fmtPct, fmtNum, fmtDollar, colorClass, etc.
- `frontend/src/utils/chartConfig.js` — Recharts shared config, heatmapColor function
- `frontend/src/components/Layout/App.jsx` — Root app with header
- `frontend/src/components/Layout/SplitView.jsx` — 40/60 split layout
- `frontend/src/components/Chat/ChatPanel.jsx` — Chat input + message list
- `frontend/src/components/Chat/MessageBubble.jsx` — Individual chat message
- `frontend/src/components/Chat/PromptSuggestions.jsx` — Example prompt chips
- `frontend/src/components/Chat/PortfolioCard.jsx` — Inline portfolio weight display in chat
- `frontend/src/components/Dashboard/ResultsPanel.jsx` — Right panel container
- `frontend/src/components/Dashboard/EquityCurve.jsx` — Recharts equity curve (log/linear toggle)
- `frontend/src/components/Dashboard/DrawdownChart.jsx` — Underwater drawdown area chart
- `frontend/src/components/Dashboard/MetricsCards.jsx` — Full metrics grid (4 sections)
- `frontend/src/components/Dashboard/MonthlyHeatmap.jsx` — Year × month returns heatmap

**Key decisions:**
- Used `cachetools.TTLCache` for price and AI response caching (thread-safe with Lock)
- Anthropic prompt caching (`cache_control: ephemeral`) on the system prompt to reduce costs
- Backtest engine chart data thinned to ≤500 points for rendering performance
- `yfinance auto_adjust=True` gives split/dividend-adjusted prices as "Close"
- Vite proxy `/api → localhost:8000` avoids CORS issues in dev

**To run locally:**
```bash
# Backend
cd backend
cp .env.example .env   # add ANTHROPIC_API_KEY
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

**Current state:** Phases 1 + 2 complete. App is runnable end-to-end.

**Pending (Phase 3+):**
- [ ] Rolling metrics chart (rolling Sharpe, vol, beta)
- [ ] Correlation matrix heatmap
- [ ] Weight drift chart
- [ ] Return distribution histogram
- [ ] Editable portfolio weights table in UI
- [ ] Benchmark overlay toggle
- [ ] Rate limiting middleware
- [ ] Mobile-responsive layout
