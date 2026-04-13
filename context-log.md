# Context Log — Portfolio Backtester

---

## 2026-04-14 — Fix event-loop blocking causing "Request failed" on production

**What:** All API routes (`/api/unified/chat`, `/api/screen/*`, `/api/backtest`) were calling blocking synchronous functions (Anthropic SDK, yfinance, pandas) directly from async FastAPI handlers. This blocks uvicorn's event loop for the full duration of the request (8–15s per request). Under load or on cold start, Railway's proxy hits its timeout and returns a non-JSON HTML error page. The frontend's `res.json()` then throws, falling back to the "Request failed" message.

**Fix:** Wrapped all blocking work in `asyncio.to_thread()` in each router. Added proper `try/except` with `HTTPException` and `logger.exception` logging to every endpoint so errors return readable JSON instead of crashing silently.

**Files modified:**
- `backend/app/routers/unified.py` — added `asyncio.to_thread` + try/except
- `backend/app/routers/screen.py` — added `asyncio.to_thread` + try/except to all 3 endpoints
- `backend/app/routers/backtest.py` — extracted `_work()` closure, runs via `asyncio.to_thread`

---

## 2026-04-14 — Drawdown tooltip bug fix + FF5 in screener rotation panel

**What:** Two fixes.

**1. Drawdown tooltip showing absurd values (e.g. -698.10%)**
Root cause: `DrawdownChart.jsx` converts raw decimal drawdown to percentage in `useMemo` (`pt.drawdown * 100`), then the tooltip called `fmtPct(value)` which multiplies by 100 again. Result was a ×10,000 exaggeration.
Fix: tooltip now renders `${value.toFixed(2)}%` directly. Also removed the now-unused `fmtPct` import. This fix covers both the main portfolio drawdown and the screener rotation panel drawdown (both use the same `DrawdownChart` component).

**2. FF5 decomposition missing from screener rotation backtest panel**
The backend already computes `ff5_decomposition` for rotation backtests (since it goes through `run_full_backtest`). Added `FamaFrenchFactors` to the `RotationPanel` component in `SplitView.jsx`, which renders the screener rotation results.

**Files modified:**
- `frontend/src/components/Dashboard/DrawdownChart.jsx` — tooltip formatting fix + removed unused import
- `frontend/src/components/Layout/SplitView.jsx` — added `FamaFrenchFactors` import + rendered after `MetricsCards` in `RotationPanel`

---

## 2026-04-13 — FF5 tooltip portal fix (all rows)

**What:** Bottom-row tooltips (Value, Profitability, Investment) were clipped by the card boundary. Rewrote `InfoTooltip` to use `ReactDOM.createPortal` — tooltip is rendered into `document.body` at a `position: fixed` coordinate calculated from the button's `getBoundingClientRect()`. Opens below the button when there's room (>160px), flips above otherwise. Left position clamped to keep it inside the viewport.

**Files modified:** `frontend/src/components/Dashboard/FamaFrenchFactors.jsx`

---

## 2026-04-13 — FF5 tooltip and significance display polish

**What:** Fixed two UI issues in the Fama-French panel.
- Tooltip was clipped at top of card — changed from `bottom: 120%` (opens upward) to `top: 120%` (opens downward) — this was a partial fix, the portal approach above was the final fix.
- Removed `***`/`**`/`*` star text from t-stat column — confusing alongside colour coding. Kept colour-only significance. Legend updated to coloured squares + p-value labels.

**Files modified:** `frontend/src/components/Dashboard/FamaFrenchFactors.jsx`

---

## 2026-04-13 — WeightDriftChart bug fixes (Y-axis, tooltip, rebalance lines)

**What:** Three separate bugs fixed in the Holdings Over Time chart.

**1. Y-axis not showing 0–100%**
AreaChart `YAxis` was missing `domain={[0, 100]}`. Added it so the stacked area always fills to 100%.

**2. Tooltip showing values like +6355%**
Root cause: chart data was already converted to percentage (`row[t] * 100`), but tooltip called `fmtPct(value)` which multiplied by 100 again. Fixed to `{p.value.toFixed(1)}%` directly.

**3. Rebalance lines missing (only 4–5 of ~40 quarterly lines visible)**
Root cause: weight history is thinned to ~300 rows server-side (step every N rows). Recharts `ReferenceLine x=` requires an exact match in the data array. Most quarterly rebalance dates were being skipped by the thinning step.
Fix: `backtest_engine.py` now explicitly preserves ALL rebalance date indices when building the thinned dataset, regardless of the step size.

**4. ReferenceLine label crash**
`label={<RebalanceLabel />}` pre-instantiates the component, preventing Recharts from injecting the `viewBox` prop → crash. Fixed to `label={RebalanceLabel}` (component reference).

**Files modified:**
- `frontend/src/components/Dashboard/WeightDriftChart.jsx` — tooltip, Y-axis domain, ReferenceLine label fix
- `backend/app/services/backtest_engine.py` — weight history thinning preserves all rebalance dates

---

## 2026-04-13 — AI chat markdown rendering fix

**What:** AI chat responses were using inline `•` separators on a single line instead of proper markdown bullet points. ReactMarkdown was already wired in `MessageBubble.jsx` correctly — the issue was the AI generating non-markdown bullet format.

**Fix:** Updated `UNIFIED_SYSTEM_PROMPT` Output Format section to explicitly require proper markdown list format (each bullet on its own line starting with `-`). Prohibited inline `•` separators.

**Files modified:** `backend/app/services/unified_ai.py`

---

## 2026-04-13 — Fama-French Five-Factor Decomposition

**What:** Added FF5 decomposition panel to every portfolio backtest result. Regresses daily portfolio excess returns on the five Fama-French factors (Mkt-RF, SMB, HML, RMW, CMA) and reports factor loadings, t-statistics, significance stars, annualised alpha, and R².

**Reference:** Fama, E.F. & French, K.R. (2015). A five-factor asset pricing model. Journal of Financial Economics, 116(1), 1–22. https://doi.org/10.1016/j.jfineco.2014.10.010

**Files created:**
- `backend/app/services/factor_decomposition.py` — OLS regression service. Primary: fetches official daily FF5 data from Ken French's data library via `pandas_datareader`. Fallback: constructs factor proxies from ETF returns (IWM-IWB for SMB, IWD-IWF for HML, QUAL-USMV for RMW, inverse MTUM for CMA). Returns alpha (annualised), betas, t-stats, significance stars, R², n_obs, source.
- `frontend/src/components/Dashboard/FamaFrenchFactors.jsx` — Table component showing all six rows (α + 5 factors). Each factor label has a `?` hover tooltip (portal-rendered, viewport-aware) with factor description and full paper citation. Significance indicated by t-stat colour only (green=p<0.001, yellow=p<0.01, pink=p<0.05, grey=n.s.). Footer repeats full citation.

**Files modified:**
- `backend/requirements.txt` — added `pandas-datareader==0.10.0`
- `backend/app/models/backtest_result.py` — added `ff5_decomposition: Optional[dict] = None` to `BacktestResult`
- `backend/app/services/backtest_engine.py` — imports `compute_ff5`, calls it after computing `portfolio_returns`, passes result into `BacktestResult`
- `backend/app/services/unified_ai.py` — added `ff5_decomposition` to sections enum in `PORTFOLIO_TOOL` and to the system prompt sections guidance (always include)
- `frontend/src/components/Dashboard/ResultsPanel.jsx` — imports `FamaFrenchFactors`, added `case 'ff5_decomposition':` in sections switch

**Decisions:**
- Alpha is annualised as `(1 + daily_alpha)^252 - 1` for comparability with CAGR
- Uses Ken French's official daily dataset when available; falls back to ETF proxies silently
- Minimum 60 observations required; returns `None` (panel hidden) if insufficient data
- `ff5_decomposition` is always added to `display_config.sections` by the AI system prompt

This file tracks all changes, decisions, and project state. Updated by Claude Code after every meaningful change.

---

## 2026-04-13 — AI Stock Screener tab (full feature)

**What:** Added a second AI mode — "AI Stock Screener" — as a new tab alongside "AI Portfolio Builder" and "Manual Build". Screener lets users ask retroactive window-based screening questions ("top return sector ETF each quarter 2022–2025"), see ranked results per window, backtest a rotation strategy, and link directly to Manual Build.

**Files created:**
- `backend/app/models/screener.py` — Pydantic models: ScreenRequest, ScreenerResult, ScreenerWindow, TickerWindowResult, RotationBacktestRequest
- `backend/app/services/screener_engine.py` — `run_screener()` (computes rank per window per metric), `run_rotation_backtest()` (no-lookahead rotation: holds previous window's winner each period)
- `backend/app/services/screener_ai.py` — AI service with SCREENER_SYSTEM_PROMPT (tool_choice=required, single shot to run_screen), concise 4-5 bullet chat reply
- `backend/app/routers/screen.py` — POST /api/screen/chat, /api/screen/run, /api/screen/backtest
- `frontend/src/components/Chat/StockScreenerPanel.jsx` — screener chat panel with suggestion chips, purple-accented
- `frontend/src/components/Dashboard/ScreenerResults.jsx` — timeline card view + full rank table, "Backtest Rotation (Top 1/3)" + "Import to Manual Build" buttons

**Files modified:**
- `backend/app/main.py` — registers screen router
- `frontend/src/styles/globals.css` — added --accent-purple, [data-mode="screener"] theme override (dark purple BG + border)
- `frontend/src/components/Layout/SplitView.jsx` — 3 tabs (Portfolio Builder / Stock Screener / Manual Build); applies data-mode="screener" to root element; splits right panel for screener (screener results left + rotation backtest right); handles rotation backtest fetch + import-to-manual flow
- `frontend/src/components/Chat/ManualBuilderPanel.jsx` — accepts screenerImport prop; useEffect auto-populates rows/settings; shows purple "Loaded from Screener" banner

**Decisions:**
- Rotation backtest uses PREVIOUS window's winner (no lookahead bias) — matches Bailey paper principles
- Color theme: screener = purple (#a855f7); portfolio = blue (#4a9eff); manual = green. Achieved via CSS data-mode attribute on root, overriding --bg-primary/secondary/card/border vars
- Screener right panel splits when rotation backtest runs: left=screener table, right=standard ResultsPanel with backtest
- Import to manual: equal-weights the top-N tickers from the LAST window; maps window_freq to rebalance_frequency
- AI screener uses tool_choice="required" (no research step needed — screener itself is the research)

---

## 2026-04-13 — Markdown rendering in chat + concise AI chat format

**What:** AI chat messages were rendering raw markdown (## headers, - bullets as plain text). Fixed by adding react-markdown to MessageBubble. Also clarified the system prompt so the chat bubble stays to a brief 3–5 bullet summary while the full analysis goes in display_config.narrative (results panel). Added "Suggested Next Steps" section to narrative template with research-paper-informed suggestions.

**Files affected:**
- `frontend/src/components/Chat/MessageBubble.jsx` — replaced `{message.content}` with `<ReactMarkdown>` for AI messages; custom component overrides maintain terminal aesthetic (▸ bullets, monospace headers, accent-blue)
- `frontend/package.json` / `package-lock.json` — added `react-markdown@10.1.0` + `remark-gfm`
- `backend/app/services/ai_service.py` — split Output Format into two sections: (1) display_config.narrative = full detailed markdown analysis in results panel; (2) final chat message = brief 3–5 bullets with a suggested next step. Prevents the full analysis wall-of-text appearing in the chat bubble.

**Decisions:**
- AiNarrative.jsx already has its own markdown parser — left untouched, no regression
- User messages and error messages rendered as plain text (no markdown parsing needed)
- react-markdown v10 uses ESM; Vite handles it natively, no config changes needed

---

## 2026-04-12 — Bailey Papers + Concise AI Output

**What:** Integrated "Deflated Sharpe Ratio" and "Probability of Backtest Overfitting" best practices from Bailey & Lopez de Prado into the backtesting and AI narrative layers. Also tightened AI output to concise bullet points.

**Files affected:**
- `backend/app/services/metrics.py` — Added `_deflated_sharpe_ratio()` helper using the DSR formula from the Bailey paper (non-normality correction via skewness + excess kurtosis, Euler-Mascheroni approximation for max expected SR). Also added `skewness` and `excess_kurtosis` to metric output.
- `backend/app/models/backtest_result.py` — Added `skewness`, `excess_kurtosis`, `deflated_sharpe` Optional[float] fields to `PortfolioMetrics`.
- `backend/app/services/ai_service.py` — Updated SYSTEM_PROMPT with a "Backtest Integrity" section (DSR thresholds: >0.95 good, 0.90–0.95 moderate, <0.90 overfit warning; overfitting risk factors; best practices to cite). Also replaced verbose narrative instructions with strict bullet-point output format. Fixed final narrative call to use cached `system` list instead of plain string.
- `frontend/src/components/Dashboard/MetricsCards.jsx` — Added DSR row (colour-coded green/yellow/red) and skewness/kurtosis rows under Risk-Adjusted section.
- `frontend/src/styles/globals.css` — Added `.warning { color: var(--accent-yellow); }` class for DSR moderate range.

**Decisions:**
- DSR uses `n_trials=1` (single strategy per request). DSR then measures significance of SR > 0 under non-normality. A more aggressive N (e.g. 50) would penalise strategies harder — deferred until user asks.
- Bailey PBO (CSCV algorithm) not implemented as it requires multiple strategy variants. Referenced in AI best practices text instead.
- DSR thresholds: >=0.95 = "low overfit risk" (green), 0.90–0.95 = "moderate risk" (yellow), <0.90 = "possible overfit" (red).

**State:** Backend computes DSR on every backtest. AI narrative now uses bullet-point format with DSR-aware overfitting warnings. Frontend displays DSR with colour coding.

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
- `frontend/src/components/Dashboard/MonthlyHeatmap.jsx` — Year x month returns heatmap

**Key decisions:**
- Used `cachetools.TTLCache` for price and AI response caching (thread-safe with Lock)
- Anthropic prompt caching (`cache_control: ephemeral`) on the system prompt to reduce costs
- Backtest engine chart data thinned to <=500 points for rendering performance
- `yfinance auto_adjust=True` gives split/dividend-adjusted prices as "Close"
- Vite proxy `/api -> localhost:8000` avoids CORS issues in dev

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

---

## 2026-04-12 — Agentic AI Loop + Dynamic Output (Phase 2 Enhancement)

**Problem solved:** The AI was making a single blind pass — guessing tickers and weights without ever seeing real data. Data-driven strategies ("most uncorrelated", "best Sharpe") were hallucinated rather than computed.

**What changed:**

**Backend:**
- `ai_service.py` — Full rewrite to agentic loop. Claude can now call `get_asset_statistics` (up to 5 iterations) before `construct_portfolio`. System prompt instructs Claude to always research before deciding on data-driven weights. Prompt caching still applied to system prompt.
- `data_service.py` — Added `get_asset_statistics_for_ai()`: computes annual return, volatility, Sharpe, max drawdown, correlation to benchmark, and full pairwise correlation matrix for a list of candidate tickers. Also added `fetch_prices_partial()` which silently drops unavailable tickers (used in research tool).
- `backtest_engine.py` — Added `_compute_rolling_metrics()`: 252d rolling Sharpe, 60d rolling volatility, 126d rolling beta. Added asset correlation matrix computation. Both included in `BacktestResult`. Data thinned to <=500 points before sending to frontend.
- `models/backtest_result.py` — Added `RollingMetrics` model, `rolling_metrics` and `correlation_matrix` fields to `BacktestResult`.
- `models/chat.py` — Added `DisplayConfig` model (sections, featured_metrics, narrative). Added `display_config` field to `ChatResponse`.
- `routers/chat.py` — Passes `display_config` from AI through to response.

**Frontend:**
- `AiNarrative.jsx` (new) — Renders AI's markdown analysis in the results panel. Supports ## headings, **bold**, bullet lists.
- `CorrelationMatrix.jsx` (new) — Asset correlation heatmap (blue=positive, red=negative).
- `RollingMetrics.jsx` (new) — Tabbed chart for rolling Sharpe / volatility / beta with configurable windows.
- `ResultsPanel.jsx` — Fully dynamic. Renders only AI-chosen sections in AI-chosen order. Shows featured metrics strip and narrative at top. Section options: equity_curve, drawdown, metrics_summary, full_metrics, monthly_heatmap, correlation_matrix, rolling_metrics.
- `useChat.js` — Added `displayConfig` state, passed through from API response.
- `SplitView.jsx` — Forwards `displayConfig` to ResultsPanel.

**Also fixed:** Upgraded yfinance from 0.2.49 -> 1.2.1 (previous version had a breaking API change with Yahoo Finance causing all downloads to fail).

**Key decisions:**
- MAX_RESEARCH_ITERATIONS = 5 to prevent infinite loops
- `get_asset_statistics` silently drops invalid tickers (`fetch_prices_partial`) so one bad ticker (e.g. VIX) doesn't block the whole research call
- display_config defaults to `["equity_curve", "drawdown", "metrics_summary"]` if AI omits it
- Rolling metrics thinned server-side to <=500 points

**Pending:**
- [ ] Return distribution histogram
- [ ] Editable portfolio weights table
- [ ] Rate limiting
- [ ] Mobile layout
- [x] Excel export

---

## 2026-04-12 — Holdings Timeseries / Weight Drift Chart

**What:** Added `WeightDriftChart` — a full timeseries of how each holding's weight evolves over the backtest period, with rebalance event markers.

**Backend changes:**
- `backtest_engine.py` — `run_backtest()` now returns `rebalance_date_strs` (list of dates when weights were reset). Weight history thinned to <=300 points before sending to frontend. Each record contains only the portfolio tickers (not benchmark).
- `models/backtest_result.py` — Added `rebalance_dates: list[str]` field to `BacktestResult`.
- `ai_service.py` — Added `weight_drift` to the `display_config.sections` enum and instructed Claude to include it for multi-asset portfolios.
- `routers/chat.py` — Updated default sections to include `weight_drift`.

**Frontend changes:**
- `WeightDriftChart.jsx` (new) — Stacked area chart (long-only) or line chart (mixed long/short) showing each holding's weight % over time. Features:
  - Auto-selects stacked area vs line based on whether short positions exist
  - Toggle between area and line mode (long-only portfolios)
  - Vertical dashed reference lines at each rebalance date
  - 10-color palette assigned per ticker
  - Custom tooltip showing all ticker weights sorted by magnitude
- `ResultsPanel.jsx` — Added `weight_drift` case to the dynamic section renderer.

**Key decisions:**
- Stacked area chart forced to line mode when shorts are present (stacking is misleading with negatives)
- Weight history thinned server-side to 300 points (enough to show drift shape without sending 2500+ rows)
- Rebalance lines drawn from index [1] onwards (skip day-0 reset which is just initialization)
- First rebalance date is always day 1 (initialization), subsequent ones are the meaningful events shown

**Pending:**
- [ ] Return distribution histogram
- [ ] Editable portfolio weights table
- [ ] Rate limiting
- [ ] Mobile layout

---

## 2026-04-13 — Bug Fixes + Screener tool_choice Fix

**What:** Fixed several bugs found during testing of the new AI Stock Screener tab and results panel.

**Bug fixes:**
- **`tool_choice={"type": "required"}` invalid** (`backend/app/services/screener_ai.py`): "required" is not a valid Anthropic API value — valid options are `"auto"`, `"any"`, or `{"type": "tool", "name": "..."}`. This caused the Anthropic API to reject the request with a non-JSON 500 error, which the frontend's `.catch()` handler converted to "Request failed". Fixed to `{"type": "any"}`.
- **Duplicate metrics table** (`frontend/src/components/Dashboard/ResultsPanel.jsx`): AI was sometimes selecting both `metrics_summary` and `full_metrics` sections. Added deduplication — filters out `metrics_summary` when `full_metrics` is also present (full_metrics is a superset).
- **Vercel deployment blocked**: Making the GitHub repo private triggered "Deployment Authorization" which blocked web-flow commits. Fixed by disabling Deployment Authorization in Vercel Settings -> Git. Build command also corrected from `vite build` to `npm run build`.

**Current state:**
- Portfolio backtester: fully working (AI chat, backtest engine, all charts, Excel export, DSR metrics, markdown chat rendering)
- AI Stock Screener: code complete; `tool_choice` bug fixed and pushed to git — will be live after Railway redeploys
- Rotation backtest: implemented with no-lookahead bias, wired to screener results panel
- Screener -> Manual Build import: wired up via `screenerImport` prop

**Pending:**
- [ ] Verify screener end-to-end after Railway redeploys with the tool_choice fix
- [ ] Return distribution histogram
- [ ] Rate limiting
- [ ] Mobile layout

---

## 2026-04-13 — Rotation Equity Chart + Port Rotation to Manual Build

**What:** Two related features: (1) visualise the timeseries of equity positions in the rotation backtest showing when each asset was bought/sold, and (2) allow the rotation strategy to be ported to the Manual Build tab for re-running and comparison.

**New files:**
- `frontend/src/components/Dashboard/RotationEquityChart.jsx` — custom equity curve for rotation backtests. Uses Recharts `ComposedChart` with:
  - Coloured `ReferenceArea` bands per window, each colour assigned to the ticker held in that window
  - Dashed `ReferenceLine` at each rotation event (window boundary)
  - Custom tooltip showing portfolio value + which ticker was held
  - Holding strip below chart: scrollable row of window cards showing label + held ticker
  - Linear/log scale toggle
  - `HoldingLegend` showing colour-to-ticker mapping

**Backend changes:**
- `backend/app/models/backtest_result.py` — added `holding_schedule: Optional[list[dict]]` field to `BacktestResult`. Each entry: `{window_start, window_end, label, tickers: [str]}`.
- `backend/app/services/screener_engine.py` — `run_rotation_backtest()` now builds and returns `holding_schedule`. Correctly handles window 0 (first_tops) and subsequent windows (rotation_map lookup with fallback).

**Frontend changes:**
- `frontend/src/components/Layout/SplitView.jsx` — major refactor of screener right panel:
  - Replaced generic `ResultsPanel` for rotation results with a new `RotationPanel` component (defined inline in SplitView)
  - `RotationPanel` shows: featured metrics strip, `RotationEquityChart`, `DrawdownChart`, `MetricsCards`, and a "→ Port to Manual Build" button
  - Added `rotationTopN` state to track which top-N was used for the rotation
  - Added `handlePortRotationToManual` — sets `rotationImport` state and switches to manual tab
  - Kept `handleImportToManual` for last-window ticker import (now distinct from rotation import)

- `frontend/src/components/Chat/ManualBuilderPanel.jsx`:
  - Added `rotationImport` prop (separate from `screenerImport`)
  - `useEffect` auto-imports rotation: pre-populates ticker rows with all unique tickers from the universe at equal weight
  - Rotation strategy banner: shows screen description, schedule table (window | held tickers), "Run Rotation" button
  - `runRotation()` calls `/api/screen/backtest` with the stored `screenResult` + `topN`
  - Results call `onResult()` so they populate the right panel with a rotation-aware strategy summary

- `frontend/src/components/Dashboard/ResultsPanel.jsx`:
  - Added import of `RotationEquityChart`
  - In `equity_curve` case: if `backtest.holding_schedule` is present and non-empty, renders `RotationEquityChart` instead of `EquityCurve` — this makes rotation results show the rich holding-band chart automatically in both the screener tab and when re-run from manual build

**Decisions:**
- Colour assignment is stable per ticker (first appearance order in holdingSchedule → palette index). Same ticker always gets same colour across chart + legend + bands.
- Port to Manual Build carries: `screenResult` (for re-running), `topN`, `holdingSchedule` (for display). Does NOT carry the computed BacktestResult (result is re-computed fresh on "Run Rotation").
- The regular ticker-row grid in Manual Build is still shown and pre-populated with the full universe at equal weight — user can modify and run a plain equal-weight backtest for comparison.
- RotationEquityChart is self-contained (no dependency on ResultsPanel) so it works both in RotationPanel (screener tab) and inside ResultsPanel (manual tab).

**Current state:**
- Rotation backtest in screener tab: shows RotationEquityChart with coloured holding bands + "Port to Manual Build" button
- Manual Build tab: supports rotation import mode with schedule table + re-run button + RotationEquityChart in results
- Both paths produce identical results (same /api/screen/backtest call)

**Pending:**
- [ ] Verify screener + rotation end-to-end after Railway redeploys
- [ ] Return distribution histogram
- [ ] Rate limiting
- [ ] Mobile layout
