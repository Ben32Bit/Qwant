# Qwant — All you ever Qwanted

An AI-powered portfolio construction, backtesting, and forecasting platform. Describe a portfolio in natural language; Qwant resolves it to real tickers, backtests it against a decade of data, runs six forecasting models in parallel, and renders a full research dashboard in seconds. No signup, no paywall.

**Live:** https://qwant.vercel.app

---

## Table of Contents
- [What Qwant does](#what-qwant-does)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Quickstart — using the live app](#quickstart--using-the-live-app)
- [Best-usage guide](#best-usage-guide)
- [Local development](#local-development)
- [Environment variables](#environment-variables)
- [API surface](#api-surface)
- [Deployment](#deployment)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

---

## What Qwant does

**Conversational portfolio construction**
- Natural-language input resolved to tickers + weights by Claude (`claude-sonnet-4`)
- Multi-step agentic tool loop: Claude can look at per-asset statistics (return, vol, Sharpe, drawdown, correlation) before committing to weights
- Full conversation context preserved — refine iteratively ("now add 10% gold", "make it more aggressive", "cut the tech exposure by half")

**Backtesting with real mechanics**
- Long + short positions (negative weights), leverage (sum > 1.0), partial cash (sum < 1.0)
- Rebalancing: daily / weekly / monthly / quarterly / annually / none (buy-and-hold drift)
- Strategy optimisers: Min Variance, Max Sharpe, Risk Parity, Inverse Volatility, Equal Weight
- Benchmark overlays: SPY, QQQ, IWM, AGG, TLT, GLD and more

**Professional-grade metrics**
- Return: Total, CAGR, best/worst year, best/worst month, monthly heatmap
- Risk: annualised vol, max drawdown + duration, downside deviation, VaR/CVaR (95/99%)
- Risk-adjusted: Sharpe, Sortino, Calmar, Information Ratio, Treynor
- Benchmark-relative: Beta, Jensen's Alpha, R², Tracking Error, Active Return, Up/Down Capture

**12-month forecast engine** (six research-backed methods, every one walk-forward OOS validated)
- **XGBoost quantile regression** — 21-day horizon, client-side ONNX inference
- **N-BEATS** — neural basis expansion, 63-day recursive forecast, client-side ONNX
- **Factor model (FF5)** — Fama-French 5-factor decomposition
- **HMM** — Hamilton (1989) regime detection over 4 regimes (bull/low-vol, bull/high-vol, bear, crisis)
- **Gaussian Process / VAR** — Sims (1980), Ledoit-Wolf shrinkage covariance
- **Attention-LSTM** — 200-pass MC dropout, browser TF.js inference
- **Regime-conditional ensemble** — Ang & Timmermann (2012) priors, disagreement-adjusted confidence bands
- **Kelly sizing** + **scenario stress tester** (soft landing, rate spike, mild recession, severe crisis, stagflation)
- **FinBERT sentiment** on SEC filings + GDELT headlines

**Live market context**
- Rolling ticker marquee: real-time prices (SPY/QQQ/major tech/commodities/crypto), GDELT headlines, Reddit trending from r/wallstreetbets, r/investing, r/stocks
- Refreshes every 5 minutes; pauses when the tab is backgrounded

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  React Frontend                     │
│            Vite + Tailwind + Recharts               │
│                                                     │
│  ┌────────────┐  ┌─────────────────────────────┐    │
│  │ Chat Panel │  │      Results Panel          │    │
│  │            │  │   (Equity, Forecast, Risk)  │    │
│  └─────┬──────┘  └──────────▲──────────────────┘    │
│        │                    │                       │
│        │         Browser-side ML:                   │
│        │         ONNX Runtime + TF.js               │
│        │         (XGBoost / N-BEATS / LSTM)         │
└────────┼────────────────────┼───────────────────────┘
         │  JSON over HTTPS   │
         ▼                    │
┌─────────────────────────────────────────────────────┐
│                FastAPI Backend                      │
│                                                     │
│  ┌──────────────┐  ┌──────────────────────────┐     │
│  │  AI Layer    │  │    Forecast Engine       │     │
│  │  (Anthropic) │  │    HMM / GP / Factor     │     │
│  └──────┬───────┘  └────────▲─────────────────┘     │
│         │                   │                       │
│  ┌──────▼───────────────────┴──────────────────┐    │
│  │     Backtest + Metrics + Optimisation       │    │
│  │     Pandas · NumPy · SciPy · scikit-learn   │    │
│  └────────────────────┬────────────────────────┘    │
│                       │                             │
│  ┌────────────────────▼────────────────────────┐    │
│  │  Data Layer: yfinance + SQLite L2 cache     │    │
│  │  Context providers: FRED, SEC EDGAR,        │    │
│  │  GDELT, Reddit, pytrends, VIX               │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

The platform is deliberately **thin-server, thick-client** for ML: XGBoost and N-BEATS ONNX weights plus the Attention-LSTM TF.js model all run in the browser. That keeps Railway's free tier well under 512MB RAM (the tf-cpu Python package alone would add ~450MB) and means forecast latency scales with user CPU, not server load.

---

## Tech stack

**Frontend**
- React 18 + Vite 6
- Tailwind CSS + Recharts
- `onnxruntime-web` (XGBoost + N-BEATS)
- `@tensorflow/tfjs` (Attention-LSTM)
- `@xenova/transformers` (FinBERT sentiment, client-side)
- `html2canvas` + `xlsx` for PNG / Excel export

**Backend**
- FastAPI + Uvicorn, Python 3.12
- `anthropic` — Claude tool-use loop
- `yfinance` + local SQLite price cache
- `pandas` / `numpy` / `scipy` — backtest + metrics + optimisation
- `hmmlearn` — Hidden Markov regime detection
- `statsmodels` — VAR for multi-asset forecasts
- `arch` — GARCH(1,1) volatility
- `scikit-learn` — Ledoit-Wolf shrinkage, feature scaling
- `fredapi` — FRED macro data · `pytrends` — search trend context

**Hosting**
- Frontend → Vercel (static)
- Backend → Railway (Docker)

---

## Quickstart — using the live app

1. Open https://qwant.vercel.app
2. Type any of these into the chat:
   - *"60/40 US stocks and bonds since 2015"*
   - *"max Sharpe portfolio of tech stocks over the last 10 years"*
   - *"risk parity across SPY, TLT, GLD, and VNQ"*
   - *"most uncorrelated 5-asset portfolio from large-cap US"*
3. Qwant resolves the description to tickers + weights, runs the backtest, and renders the full dashboard.
4. Refine in plain English: *"add 10% gold"*, *"change to monthly rebalancing"*, *"benchmark against QQQ instead"*.
5. Click **Run Forecast** in the Forecast panel to kick off the six-model 12-month projection.
6. Export as PNG (full forecast sheet including scenario tester) or Excel (all metrics + equity curve).

---

## Best-usage guide

Qwant is most useful as a **research co-pilot**, not a trade executor. Here's how to get the most out of it.

### 1. Start broad, then refine
The conversational loop is its strongest feature. Instead of hand-tuning weights, describe the **intent** and let the AI's research tools look at real data first:

> "I want a defensive portfolio of 5 assets that should survive a stagflation scenario."

Claude will call `get_asset_statistics` internally — looking at historical returns, vol, Sharpe, and correlations — before committing. You'll often get a better starting point than hand-picking, because it can see the data you haven't looked at yet.

Then refine:
- *"drop anything with correlation above 0.5 to SPY"*
- *"cap any single position at 25%"*
- *"now tilt it toward value — swap in VTV"*

### 2. Pick the right optimiser for the question
| Goal | Strategy to request |
|------|---------------------|
| "Safest mix of these assets" | `min_variance` |
| "Best risk-adjusted return" | `max_sharpe` |
| "Equal risk contribution from every asset" | `risk_parity` |
| "Weight by the inverse of volatility" | `inverse_vol` |
| "Naive 1/N baseline" | `equal_weight` |
| "Use my exact weights" | `custom` (default if you specify weights) |

Say *"use max Sharpe"* or *"apply risk parity"* in chat — the AI picks the right tool.

### 3. Benchmarks matter more than you think
Change the benchmark based on the portfolio:
- **Equity-heavy** → `SPY` (default) or `QQQ` for tech tilt
- **Global** → `VT`, `EFA`, or `ACWI`
- **Bond-heavy** → `AGG` or `BND`
- **Alternatives-heavy** → try an equal-weight mix by comparing against `VTI` + overlay

Alpha, beta, and Up/Down capture are only meaningful against the right benchmark.

### 4. Read the forecast like a research analyst, not a crystal ball
- **Honest horizon caps**: XGBoost's fan band stops at 21 days, N-BEATS at 63 days. Don't stare at the right-hand side of the 12m exploratory chart — the model count strip at the bottom tells you how many methods are still in play at each point.
- **Regime probabilities** matter more than the point forecast. If the HMM says 60% bear / 20% crisis, the ensemble is going to lean conservative — that's signal.
- **Disagreement widens confidence bands**: when the six methods disagree, the ensemble fan expands. Tight fan = all methods agree. Wide fan = "nobody actually knows".
- **Scenario tester** > point forecast for risk decisions. Run the rate-spike and severe-crisis scenarios to see ensemble drift before making a real bet.

### 5. Rebalancing: trade off turnover vs drift
- **Quarterly** is the default and the best general-purpose choice.
- **Monthly** for tactical portfolios where you want the weights to stay tight.
- **Annually** if you care about tax drag or want to see how buy-and-hold drift diverges.
- **None** for pure buy-and-hold — useful for isolating the return contribution from an individual outlier asset.

Compare the same portfolio at different frequencies to see how rebalance frequency affects Sharpe.

### 6. Short positions and leverage — use them deliberately
- Short: negative weight (e.g. `AAPL: -0.20` = 20% short). Good for pair trades or hedged long books.
- Leverage: weights summing above 1.0 (e.g. total 1.5 = 150% exposure). Returns and drawdowns are amplified; **read the max drawdown carefully**.
- Qwant does not currently charge a borrow cost for leveraged positions; results are therefore slightly optimistic for 2x+ portfolios. Adjust expectations manually.

### 7. Use the market-context ticker for macro framing
The rolling ticker at the top is not decoration. When you're about to commit to a long-duration bond sleeve and the top headline is *"Fed signals three more hikes"*, that's the moment to stress-test against the rate-spike scenario before shipping a recommendation.

### 8. Export when you're done
- **PNG** exports the full forecast sheet including the scenario tester — good for sharing with a collaborator.
- **Excel** dumps equity curve + every metric to a spreadsheet — good for downstream modelling.

Both are lazy-loaded; the libraries only download when you click the button.

### 9. What Qwant is NOT for
- Intraday trading decisions (daily-bar backtests only)
- Tax-optimised portfolio construction (no account for wash sales, lot selection, tax lots)
- Options / derivatives (equities and ETFs only)
- Emerging-market local-listed equities (US-listed ADRs work; raw local tickers don't)
- Live trading — there is no broker connection by design

### 10. The free tier is rate-limited
- 15 requests / hour / IP for AI-backed endpoints
- 200 requests / hour / IP globally
- Hit the limit? Wait it out or run locally (see below).

---

## Local development

### Prerequisites
- Python 3.12
- Node 20+
- An Anthropic API key (`ANTHROPIC_API_KEY`)

### Backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env                     # fill in ANTHROPIC_API_KEY
uvicorn app.main:app --reload --port 8000
```

The backend warms a 50-ticker, 10-year SQLite price cache on first boot (~8 MB, fire-and-forget — the app is live while it fills).

### Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

By default the dev server talks to `http://localhost:8000`. Set `VITE_API_BASE` in `frontend/.env.local` to point at a different host.

### Run the production build locally

```bash
cd frontend
npm run build
npm run preview       # serves the built dist/ on :4173
```

---

## Environment variables

### Backend (`backend/.env`)

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | _required_ | Claude API key for the chat endpoint |
| `RISK_FREE_RATE` | `0.05` | Used for Sharpe / Sortino / Treynor |
| `CACHE_TTL_HOURS` | `24` | Historical price cache TTL |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated list |
| `AI_RATE_LIMIT` | `10/hour;50/day` | Chat endpoint rate limit |
| `FORECAST_RATE_LIMIT` | `12/hour;40/day` | Forecast endpoint rate limit |
| `COMPUTE_RATE_LIMIT` | `30/hour;120/day` | Direct `/backtest` rate limit |
| `GLOBAL_RATE_LIMIT` | `200/hour` | App-wide fallback |
| `FRED_API_KEY` | _optional_ | FRED macro data (auto-skips if unset) |

### Frontend (`frontend/.env.local`)

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE` | `http://localhost:8000` | Backend URL |

---

## API surface

All routes under `/api`. Rate-limited per-IP (no auth).

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | Conversational portfolio construction + backtest |
| `POST` | `/api/backtest` | Direct backtest (bypass AI, manual weights) |
| `POST` | `/api/forecast` | Run HMM / GP / Factor / VAR forecast pass |
| `GET` | `/api/tickers/search?q=...` | Autocomplete ticker search |
| `GET` | `/api/benchmarks` | List supported benchmark ETFs |
| `GET` | `/api/ticker/feed` | Rolling marquee: prices + GDELT + Reddit |
| `POST` | `/api/screen` | Stock screener (fundamentals + technicals) |
| `POST` | `/api/unified` | Combined AI chat + screener |
| `GET` | `/api/regime` | Current macro regime classification |
| `GET` | `/health` | Liveness probe |
| `GET` | `/api/debug/memory` | RSS memory snapshot |

Full request / response shapes live in `CLAUDE.md` → "API Endpoints".

---

## Deployment

**Frontend (Vercel)**: push to `master`; Vercel auto-builds `frontend/` and deploys. Static. Free tier is fine.

**Backend (Railway)**: Docker build from `backend/Dockerfile`. Free tier holds if the warm-cache completes successfully. Watch `/api/debug/memory` — target is <400MB RSS. If you exceed 512MB you'll be evicted.

Tensorflow was explicitly removed from the backend and moved to browser TF.js to stay under the Railway RAM cap — don't add it back without moving to a paid tier.

---

## Project layout

```
Qwant/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat/           # ChatPanel, MessageBubble, StockScreenerPanel
│   │   │   ├── Dashboard/      # EquityCurve, ForecastPanel, DrawdownChart, …
│   │   │   ├── Layout/         # App, SplitView, TickerBar
│   │   │   └── Portfolio/      # PortfolioTable, WeightBar
│   │   ├── hooks/              # useBacktest, useChat, useForecast
│   │   ├── ml/                 # XGBoostInferer, NBeatsInferer, LSTMInferer, MetaEnsemble
│   │   ├── utils/              # formatters, chartConfig, exportExcel, api
│   │   └── styles/globals.css
│   ├── public/                 # favicon, ONNX model weights, LSTM tf.js model
│   └── vite.config.js
│
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI app, CORS, lifespan
│   │   ├── routers/            # chat, backtest, forecast, ticker, screen, unified
│   │   ├── services/
│   │   │   ├── ai_service.py         # Anthropic tool-use loop
│   │   │   ├── backtest_engine.py    # Core backtest math
│   │   │   ├── forecast_engine.py    # HMM / GP / VAR / Factor models
│   │   │   ├── metrics.py            # All portfolio metric calcs
│   │   │   ├── optimization.py       # Min var, max Sharpe, risk parity
│   │   │   ├── data_service.py       # yfinance + SQLite L2 cache
│   │   │   ├── price_store.py        # Shared price cache
│   │   │   └── {fred,sec,news,reddit,vix,trends}_provider.py
│   │   ├── models/             # Pydantic schemas
│   │   └── utils/
│   ├── scripts/                # Training scripts (ONNX exports)
│   └── requirements.txt
│
├── CLAUDE.md                   # Design spec + full API reference (internal)
├── context-log.md              # Change log per session
└── README.md
```

---

## Troubleshooting

**"Forecast phase 2 timed out"** — Railway backend is cold-starting. First forecast on a fresh dyno can take 30-60s; subsequent runs are cached. Retry once.

**"Loading market feed…" never resolves** — the `/api/ticker/feed` endpoint couldn't reach GDELT or Reddit (both are unauthenticated public APIs and do rate-limit). The marquee fails open; other features are unaffected.

**Tab freezes after forecast** — known issue, fixed in commit `5344123`. If you're running an older build, pull latest.

**ONNX 404s in the console** (`bull_low_vol.onnx`) — optional meta-learner stacker that is not shipped in the public build. The app falls back to the Ang & Timmermann rule-based weights automatically. Safe to ignore.

**SharedArrayBuffer warning** — ONNX Runtime Web wants cross-origin isolation headers (COOP/COEP) for multi-threaded WASM. Without them it runs single-threaded, which is fine.

**Rate limited in chat** — you've hit the 10/hour AI limit. Wait it out, or run the backend locally with a higher `AI_RATE_LIMIT`.

**Backtest returns empty equity curve** — ticker probably doesn't have data for the requested date range. Try the exchange ticker form: `BRK-B` not `BRK.B`; `.L` suffix for LSE ADRs, etc.

---

## Acknowledgements

Research references baked into the forecasting engine:
- Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* Econometrica.
- Ang, A. & Timmermann, A. (2012). *Regime Changes and Financial Markets.* Annu. Rev. Financ. Econ.
- Fama, E.F. & French, K.R. (2015). *A five-factor asset pricing model.* J. Financ. Econ.
- Ledoit, O. & Wolf, M. (2004). *Honey, I Shrunk the Sample Covariance Matrix.* J. Portf. Manag.
- Oreshkin, B.N. et al. (2020). *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.* ICLR.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
