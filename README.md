# Qwant — All you ever Qwanted

An AI-powered portfolio construction, backtesting, and forecasting platform. Describe a portfolio in natural language; Qwant resolves it to real tickers, backtests it against a decade of data, runs five probabilistic forecasting models in parallel, and renders a full research dashboard in seconds. No signup, no paywall.

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
- [Acknowledgements](#acknowledgements)

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

**3-month probabilistic forecast engine** (five research-backed methods, all walk-forward OOS validated)

Every method goes through a **shadow holdout evaluation** before contributing to the live forecast: it trains on data up to T−60, forecasts the window T−60 → T−30, and the accuracy of that forecast (OOS R²) drives its ensemble weight. Methods with OOS R² < −0.5 are excluded entirely.

| Method | Where | OOS R² |
|--------|-------|--------|
| **N-BEATS** — 3 residual stacks, 63-day recursive horizon, 5 quantiles (pinball loss) | Browser (pure-JS) | Shadow holdout |
| **TimesFM 2.5** — Google 200M-parameter decoder-only foundation model, zero-shot | Server (always-warm) | Shadow holdout |
| **HMM** — Hamilton (1989) 2-state Baum-Welch, 1 000-path regime-conditional simulation | Server | Shadow holdout |
| **Gaussian Process** — Matérn ν=5/2 ARD kernel, exact Bayesian uncertainty, 63-day rollout | Server | Shadow holdout |
| **Attention-LSTM** — Bahdanau-attention LSTM(64), MC Dropout 200 passes (TF.js) | Browser | Shadow holdout |

**OOS R²-weighted ensemble** (Wolpert 1992 stacked generalisation)
- Shadow OOS R² → clip negative to 0 → normalise → TimesFM 30% floor → blend p5/p25/p50/p75/p95
- Disagreement across methods widens outer bands (Krogh & Vedelsby 1995)
- HMM × VIX → 4-state regime display (bull/bear × vol) shown in donut — **not** used for weighting

**Position sizing & stress testing**
- Kelly sizing: f\* = μ/σ² · half-Kelly · regime confidence adjustment
- Scenario stress tester: 6 macro regimes (soft landing, rate spike, mild recession, severe crisis, stagflation, reflation)
- FinBERT sentiment on SEC filings + GDELT headlines (browser NLP)

**Live market context**
- Rolling ticker marquee: real-time prices (SPY/QQQ/major tech/commodities/crypto), English-filtered GDELT headlines, Reddit trending from r/wallstreetbets, r/investing, r/stocks
- Refreshes every 60 seconds; pauses when the tab is backgrounded

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   React Frontend                     │
│             Vite + Tailwind + Recharts               │
│                                                      │
│  ┌────────────┐   ┌──────────────────────────────┐   │
│  │ Chat Panel │   │      Results Panel           │   │
│  │            │   │  (Equity, Forecast, Risk)    │   │
│  └─────┬──────┘   └──────────▲───────────────────┘   │
│        │                     │                       │
│        │          Browser-side ML:                   │
│        │          N-BEATS (pure-JS) + LSTM (TF.js)   │
│        │          FinBERT (Xenova/transformers)       │
│        │          MetaEnsemble (OOS R²-weighted)      │
└────────┼─────────────────────┼───────────────────────┘
         │   JSON over HTTPS   │
         ▼                     │
┌──────────────────────────────────────────────────────┐
│                  FastAPI Backend                     │
│                                                      │
│  ┌───────────────┐  ┌──────────────────────────┐     │
│  │   AI Layer    │  │     Forecast Engine      │     │
│  │  (Anthropic)  │  │  HMM · GP · TimesFM 2.5  │     │
│  └──────┬────────┘  └───────▲──────────────────┘     │
│         │                   │                        │
│  ┌──────▼───────────────────┴─────────────────────┐  │
│  │    Backtest + Metrics + Optimisation           │  │
│  │    Pandas · NumPy · SciPy · scikit-learn       │  │
│  └───────────────────┬─────────────────────────┬──┘  │
│                      │                         │     │
│  ┌───────────────────▼──────┐  ┌───────────────▼──┐  │
│  │  yfinance + price cache  │  │  Context providers│  │
│  │  (SQLite L2, 10y daily)  │  │  FRED · VIX ·    │  │
│  └──────────────────────────┘  │  SEC · GDELT ·   │  │
│                                │  Reddit · EDGAR  │  │
│                                └──────────────────┘  │
└──────────────────────────────────────────────────────┘
```

**Thin-server, thick-client for ML**: N-BEATS (pure-JS matrix math) and the Attention-LSTM (TF.js) run entirely in the browser. TimesFM 2.5 (200M parameters, PyTorch) runs server-side as an always-warm singleton loaded at startup — it's too large for the browser but benefits from Railway's 8 GB RAM ceiling.

The LSTM is **retrained weekly** via GitHub Actions (`.github/workflows/retrain-lstm.yml`, Sunday 06:00 UTC). The job pulls fresh price data, trains with Huber loss + scale-invariant volatility-normalised features, and commits the new TF.js artefacts only if the validation gate passes (MSE / var(y_test) ≤ 1.10 on a rolling 12-month held-out window).

---

## Tech stack

**Frontend**
- React 18 + Vite 6
- Tailwind CSS + Recharts
- `@tensorflow/tfjs` — Attention-LSTM MC Dropout inference
- `@xenova/transformers` — FinBERT sentiment, fully client-side
- `onnxruntime-web` — optional ONNX meta-stacker (disabled in production; `META_ONNX_AVAILABLE = false`)
- `html-to-image` + `xlsx` for PNG / Excel export

**Backend**
- FastAPI + Uvicorn, Python 3.12
- `anthropic` — Claude tool-use loop (`claude-sonnet-4`)
- `torch` + `transformers >= 4.48` — TimesFM 2.5 foundation model (CPU wheel, ~200 MB)
- `yfinance` + local SQLite price cache
- `pandas` / `numpy` / `scipy` — backtest, metrics, optimisation
- `hmmlearn` — 2-state Hidden Markov regime detection (Baum-Welch EM)
- `scikit-learn` — Ledoit-Wolf covariance shrinkage, feature scaling
- `fredapi` / FRED REST API — macro data (T10Y2Y, BAA10Y, DFII10, DFF)
- `cachetools` — TTLCache for TimesFM inference results

**Hosting**
- Frontend → Vercel (static)
- Backend → Railway (Docker, CPU-only PyTorch wheel)

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
5. Click **Run Forecast** in the Forecast panel to kick off the five-model 3-month projection.
6. Export as PNG (full forecast sheet including scenario tester) or Excel (all metrics + equity curve).

---

## Best-usage guide

Qwant is most useful as a **research co-pilot**, not a trade executor. Here's how to get the most out of it.

### 1. Start broad, then refine
The conversational loop is its strongest feature. Instead of hand-tuning weights, describe the **intent** and let the AI's research tools look at real data first:

> "I want a defensive portfolio of 5 assets that should survive a stagflation scenario."

Claude will call `get_asset_statistics` internally — looking at historical returns, vol, Sharpe, and correlations — before committing. You'll often get a better starting point than hand-picking.

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

### 3. Benchmarks matter more than you think
Change the benchmark based on the portfolio:
- **Equity-heavy** → `SPY` (default) or `QQQ` for tech tilt
- **Global** → `VT`, `EFA`, or `ACWI`
- **Bond-heavy** → `AGG` or `BND`

Alpha, beta, and Up/Down capture are only meaningful against the right benchmark.

### 4. Read the forecast like a research analyst, not a crystal ball
- **OOS R² drives everything**: each method is tested on a 30-day shadow holdout before the live forecast runs. Methods that score OOS R² < −0.5 (errors 50%+ worse than the historical mean) are excluded from the composite chart and shown with an explanation on their card.
- **IS R² vs OOS R²**: the card pair tells you in-sample fit vs genuine out-of-sample accuracy. High IS R² with low OOS R² = overfitting.
- **Disagreement widens confidence bands**: when the five methods disagree, the ensemble fan expands. Tight fan = consensus. Wide fan = genuine uncertainty.
- **Shadow window (amber region)**: the shaded band on the composite chart is the OOS test window. Each model's dashed line is its holdout forecast; the blue portfolio line shows what actually happened. The gap is what OOS R² measures.
- **Scenario tester** > point forecast for risk decisions. Run the rate-spike and severe-crisis scenarios to see ensemble drift before making a real bet.

### 5. Rebalancing: trade off turnover vs drift
- **Quarterly** is the default and best general-purpose choice.
- **Monthly** for tactical portfolios where you want tight weights.
- **Annually** to see buy-and-hold drift vs rebalanced.
- **None** for pure buy-and-hold — useful for isolating single-asset contribution.

### 6. Short positions and leverage — use deliberately
- Short: negative weight (e.g. `AAPL: -0.20` = 20% short). Good for pair trades or hedged long books.
- Leverage: weights summing above 1.0 (e.g. total 1.5 = 150% exposure). Returns and drawdowns are amplified; **read the max drawdown carefully**.
- Qwant does not currently charge a borrow cost for leveraged positions; results are therefore slightly optimistic for 2x+ portfolios.

### 7. Use the market-context ticker for macro framing
The rolling ticker at the top is not decoration. When you're about to commit to a long-duration bond sleeve and the top headline is *"Fed signals three more hikes"*, that's the moment to stress-test against the rate-spike scenario.

### 8. Export when you're done
- **PNG** exports the full forecast sheet including the scenario tester — good for sharing.
- **Excel** dumps equity curve + every metric to a spreadsheet — good for downstream modelling.

Both are lazy-loaded; the libraries only download when you click the button.

### 9. What Qwant is NOT for
- Intraday trading decisions (daily-bar backtests only)
- Tax-optimised portfolio construction
- Options / derivatives (equities and ETFs only)
- Emerging-market local-listed equities (US-listed ADRs work; raw local tickers don't)
- Live trading — there is no broker connection by design

### 10. The free tier is rate-limited
- 10 requests / hour / IP for AI chat
- 12 requests / hour / IP for forecasts
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

The backend warms a 50-ticker, 10-year price cache on first boot (~8 MB, fire-and-forget). TimesFM 2.5 loads its 200M weights into RAM (~200 MB) in the background — the first forecast request after a cold boot may add ~30s while it downloads.

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
| `FRED_API_KEY` | _optional_ | FRED macro data (uses anonymous pandas_datareader fallback if unset) |

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
| `POST` | `/api/forecast` | Run HMM / GP / TimesFM / N-BEATS features / LSTM features |
| `POST` | `/api/regime/current` | Current 4-state regime classification from live equity curve |
| `GET` | `/api/tickers/search?q=...` | Autocomplete ticker search |
| `GET` | `/api/benchmarks` | List supported benchmark ETFs |
| `GET` | `/api/ticker/feed` | Rolling marquee: prices + GDELT + Reddit |
| `POST` | `/api/screen` | Stock screener (fundamentals + technicals) |
| `POST` | `/api/unified` | Combined AI chat + screener |
| `GET` | `/health` | Liveness probe |
| `GET` | `/api/debug/memory` | RSS memory snapshot |

Full request / response shapes live in `CLAUDE.md` → "API Endpoints".

---

## Deployment

**Frontend (Vercel)**: push to `master`; Vercel auto-builds `frontend/` and deploys. Static. Free tier is fine.

**Backend (Railway)**: Docker build from `backend/Dockerfile`. TimesFM 2.5 adds ~200 MB RAM on top of the Python baseline; target is <600 MB RSS total. Watch `/api/debug/memory`. The CPU-only PyTorch wheel (via `--extra-index-url https://download.pytorch.org/whl/cpu`) keeps the Docker image ~1.5 GB smaller than the CUDA build.

Do not add TensorFlow to the backend — TF-CPU alone is ~450 MB and would push Railway into OOM on the Hobby plan.

---

## Project layout

```
Qwant/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat/           # ChatPanel, MessageBubble, StockScreenerPanel
│   │   │   ├── Dashboard/      # EquityCurve, ForecastPanel, ForecastComposite,
│   │   │   │                   # ForecastMethodCard, ForecastArchitecture,
│   │   │   │                   # EnsembleCard, KellyPanel, ScenarioPanel, …
│   │   │   ├── Layout/         # App, SplitView, TickerBar
│   │   │   └── Portfolio/      # PortfolioTable, WeightBar
│   │   ├── hooks/              # useBacktest, useChat, useForecast
│   │   ├── ml/
│   │   │   ├── NBeatsInferer.js    # Pure-JS N-BEATS rollout
│   │   │   ├── LSTMInferer.js      # TF.js MC Dropout inference + IS/OOS R²
│   │   │   ├── MetaEnsemble.js     # OOS R²-weighted blend, disagreement bands
│   │   │   ├── KellyCalculator.js  # Half-Kelly position sizing
│   │   │   └── SentimentInferer.js # FinBERT (Xenova)
│   │   ├── utils/              # formatters, chartConfig, exportExcel, api
│   │   └── styles/globals.css
│   ├── public/
│   │   └── models/
│   │       └── lstm/           # TF.js model weights (retrained weekly)
│   └── vite.config.js
│
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI app, CORS, lifespan (warms TimesFM)
│   │   ├── routers/            # chat, backtest, forecast, ticker, screen, regime
│   │   ├── services/
│   │   │   ├── ai_service.py           # Anthropic tool-use agentic loop
│   │   │   ├── backtest_engine.py      # Core backtest math + FF5 decomposition
│   │   │   ├── forecast_engine.py      # HMM · GP · TimesFM · N-BEATS/LSTM features
│   │   │   │                           # Shadow holdout OOS R² evaluation
│   │   │   ├── timesfm_provider.py     # TimesFM 2.5 singleton + TTLCache
│   │   │   ├── meta_learner.py         # OOS R²-weighted ensemble weights
│   │   │   ├── metrics.py              # All portfolio metric calculations
│   │   │   ├── optimization.py         # Min var, max Sharpe, risk parity
│   │   │   ├── data_service.py         # yfinance + SQLite L2 cache
│   │   │   ├── factor_decomposition.py # Fama-French 5-factor OLS (Ken French data)
│   │   │   └── {fred,sec,vix,news,reddit,edgar}_provider.py
│   │   ├── models/             # Pydantic schemas (portfolio, backtest, regime)
│   │   └── utils/
│   ├── scripts/
│   │   └── train_lstm.py       # Weekly LSTM training + TF.js export
│   └── requirements.txt
│
├── .github/
│   └── workflows/
│       └── retrain-lstm.yml    # Weekly LSTM retraining (Sunday 06:00 UTC)
│
├── CLAUDE.md                   # Design spec + full API reference (internal)
├── context-log.md              # Per-session change log
└── README.md
```

---

## Troubleshooting

**"Forecast phase 2 timed out after 180s"** — Railway backend is cold-starting or TimesFM 2.5 is loading its weights for the first time (~30s on cold boot). The client retries once automatically on 502/504. If the retry also fails, click Re-run Forecast — the container will be warm.

**"Loading market feed…" never resolves** — `/api/ticker/feed` couldn't reach GDELT or Reddit (both are unauthenticated public APIs that rate-limit). The marquee fails open; all other features are unaffected.

**LSTM OOS R² is missing on the card** — the shadow evaluation runs after the main 200-pass rollout in the browser. It requires the `shadow` object in `lstm_features` (server must send it). If the LSTM card shows `—` for OOS R², the history is too short for a shadow window (need ≥150 trading days).

**SharedArrayBuffer warning** — ONNX Runtime Web wants COOP/COEP cross-origin isolation headers for multi-threaded WASM. Without them it runs single-threaded, which is fine for the optional meta-stacker (currently disabled anyway, `META_ONNX_AVAILABLE = false`).

**Rate limited in chat** — you've hit the 10/hour AI limit. Wait it out, or run the backend locally with a higher `AI_RATE_LIMIT`.

**Backtest returns empty equity curve** — ticker probably has no data for the requested date range. Try the exchange-qualified form: `BRK-B` not `BRK.B`; append `.L` for LSE listings, `.TO` for TSX, etc.

**"TimesFM model is not available"** — Railway failed to download model weights at startup (network issue or HuggingFace hub outage). The TimesFM card will show an error; all other methods are unaffected. Re-deploy the Railway service to trigger a fresh boot.

---

## Acknowledgements

Research references baked into the platform:

**Forecast methods**
- Oreshkin, B.N. et al. (2020). *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.* ICLR.
- Das, A., Kong, W., Leach, A. et al. (2024). *A decoder-only foundation model for time-series forecasting.* ICML. (TimesFM 2.5)
- Hamilton, J.D. (1989). *A new approach to the economic analysis of nonstationary time series and the business cycle.* Econometrica, 57(2).
- Rasmussen, C.E. & Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning.* MIT Press. (GP Matérn ν=5/2)
- Bahdanau, D., Cho, K. & Bengio, Y. (2015). *Neural machine translation by jointly learning to align and translate.* ICLR. (LSTM attention)
- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian approximation.* ICML. (LSTM MC Dropout)

**Ensemble & uncertainty**
- Wolpert, D.H. (1992). *Stacked generalization.* Neural Networks, 5(2), 241–259.
- Krogh, A. & Vedelsby, J. (1995). *Neural network ensembles, cross validation, and active learning.* NeurIPS 8.
- Koenker, R. & Bassett, G. (1978). *Regression quantiles.* Econometrica, 46(1). (N-BEATS pinball loss)

**Validation & sizing**
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning,* Ch. 7 (walk-forward CV). Wiley.
- Kelly, J.L. (1956). *A new interpretation of information rate.* Bell System Technical Journal, 35(4).
- Ang, A., Hodrick, R.J., Xing, Y. & Zhang, X. (2006). *The cross-section of volatility and expected returns.* Journal of Finance, 61(1). (VIX regime)

**Portfolio construction**
- Ledoit, O. & Wolf, M. (2004). *Honey, I shrunk the sample covariance matrix.* Journal of Portfolio Management.
- Fama, E.F. & French, K.R. (2015). *A five-factor asset pricing model.* Journal of Financial Economics.
