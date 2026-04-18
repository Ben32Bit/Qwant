# Context Log — Portfolio Backtester

---

## 2026-04-19 — Fix: rate_limit.py startup crash (list vs string)

**Root cause:** `os.getenv(...).split(";")` returned a Python list (`["10/hour", "50/day"]`). slowapi 0.1.9's `@limiter.limit()` decorator expects a plain string, not a list — it called string methods on the list at decoration time, raising `TypeError` before uvicorn could bind, so Railway's healthcheck at `/health` never got a response.

**Fix:** Removed `.split(";")` from `rate_limit.py`; limits are now plain strings (`"10/hour"` etc.). Daily caps removed for now — the hourly limit is the primary bot protection and is sufficient. Updated `.env.example` to match.

**Files:** `backend/app/utils/rate_limit.py`, `backend/.env.example`

---

## 2026-04-19 — Rate limiting hardened across all endpoints

**What:** Every API endpoint now has per-IP rate limits to protect Anthropic token spend when sharing the app publicly.

**Problem:** Only `/api/chat` and `/api/backtest` had limits (20/h and 60/h respectively). `/api/unified/chat`, `/api/screen/chat`, `/api/forecast`, `/api/screen/run`, `/api/screen/backtest`, and `/api/regime/current` had **zero** rate limiting — a bot could exhaust the Anthropic quota instantly.

**Solution:** Consolidated into a single shared limiter (`app/utils/rate_limit.py`) with three tiers:

| Tier | Endpoints | Default limit | Rationale |
|------|-----------|--------------|-----------|
| `AI_LIMITS` | `/chat`, `/unified/chat`, `/screen/chat` | **10/hour · 50/day** | Each call costs ~$0.003–0.01 in Anthropic tokens; 50/day ≈ $0.25–0.50 max per IP |
| `FCST_LIMITS` | `/forecast` | **12/hour · 40/day** | 2 server calls per full forecast run (phase 1 + phase 2) → 6 full runs/hour |
| `CMPT_LIMITS` | `/backtest`, `/screen/run`, `/screen/backtest`, `/regime/current` | **30/hour · 120/day** | No Anthropic cost, but CPU-intensive |
| Global | All routes | **200/hour** | Safety net via `default_limits` on the Limiter |

All limits are env-configurable (`AI_RATE_LIMIT`, `FORECAST_RATE_LIMIT`, `COMPUTE_RATE_LIMIT`, `GLOBAL_RATE_LIMIT`). slowapi sends `X-RateLimit-*` headers so the client can see remaining quota.

**Files changed:**
- `backend/app/utils/rate_limit.py` — new shared limiter module
- `backend/app/main.py` — import shared limiter, remove stale `CHAT_RATE`/`BACKTEST_RATE` vars
- `backend/app/routers/chat.py` — use shared limiter + AI_LIMITS
- `backend/app/routers/backtest.py` — use shared limiter + CMPT_LIMITS (was 60/h, now 30/h + daily cap)
- `backend/app/routers/unified.py` — add `Request` param + AI_LIMITS (was completely unprotected)
- `backend/app/routers/screen.py` — add `Request` params + AI_LIMITS on /screen/chat, CMPT_LIMITS on /screen/run + /screen/backtest
- `backend/app/routers/forecast.py` — add FCST_LIMITS (was completely unprotected)
- `backend/app/routers/regime.py` — add CMPT_LIMITS (was completely unprotected)
- `backend/.env.example` — document all four new env vars

---

## 2026-04-19 — Forecast reliability fixes + method effectiveness UI (session 2)

**Commits:** `40079cd`, `73e30a7`, `823457e`, `5c441de`, `3533d57`, `9ccfea6`

### Fix: LSTM always runs regardless of phase 2 outcome (`40079cd`)
- Root cause: `if (!p2Data) return` in `useForecast.js` meant LSTM was silently skipped whenever phase 2 timed out or errored.
- Fix: Added `deriveLstmFeatures(equityCurve)` helper that builds the seed window (last 60 scaled daily returns), scaler min/max, and 252 business-day forecast dates from the equity curve directly.
- LSTM now always runs after phase 2 — using server features when available, client-derived features as fallback.
- Added `lstmStandalone` state: when `p2Data` is null, LSTM result stored separately and merged into `allResults` via updated `mergeResults(phase1, phase2, lstmStandalone)`.
- Phase 2 timeout reduced from 90s → 25s (HMM/GP should complete in <15s; longer means Railway is struggling).
- **Files:** `frontend/src/hooks/useForecast.js`

### Feature: Method effectiveness table in composite chart (`40079cd`)
- Replaced the simple ensemble weight strip (colored bar) with `MethodEffectivenessTable`.
- Shows all 6 methods in a grid: colored dot + name, ensemble weight % with animated bar, OOS quality badge (HIGH/MED/LOW derived from whichever metric the method exposes: `oos_r2` for XGBoost, `oos_mse` for LSTM, `regime_sanity` for HMM, `ljung_box_ok` for GP/VAR), 12-month p50 median return endpoint.
- Footer note cites Wolpert (1992) stacked generalisation.
- **Files:** `frontend/src/components/Dashboard/ForecastComposite.jsx`

### Fix: Results tab ETA bar estimate (`73e30a7`, `40079cd`)
- `BacktestEtaBar` estimate raised 12s → 40s (AI + backtest pipeline regularly takes 20–35s).
- **Files:** `frontend/src/components/Dashboard/ResultsPanel.jsx`

### Fix: ETA bar stuck at "~0s" when overrunning (`73e30a7`)
- When `elapsed > estMs`, `etaMs = 0` caused the bar to freeze at "~0s".
- Fixed by detecting overrun and switching to amber `⚠ +Xs — still running` display + amber/orange gradient on the progress bar.
- Added `BacktestEtaBar` to `ResultsPanel` (tracks `loading` prop with `loadingStartRef`, 40s estimate, overrun state).
- **Files:** `frontend/src/components/Dashboard/ForecastPanel.jsx`, `frontend/src/components/Dashboard/ResultsPanel.jsx`

### Fix: Composite chart showing only N-BEATS (`5c441de`)
- `METHOD_ORDER` in `ForecastComposite` was `['monte_carlo', 'garch', ...]` — stale names from a previous engine version. No XGBoost or Factor lines were ever rendered.
- Fixed to `['xgboost', 'nbeats', 'factor', 'hmm', 'var', 'lstm']`.
- Also reordered `ForecastPanel`: composite chart + 6 method cards first (primary outputs), then EnsembleCard / KellyPanel / ScenarioPanel / SentimentPanel (meta-analysis).
- **Files:** `frontend/src/components/Dashboard/ForecastComposite.jsx`, `frontend/src/components/Dashboard/ForecastPanel.jsx`

### Feature: Phase 8 — Macro scenario stress tester (`3533d57`)
- `ScenarioPanel.jsx`: 6 macro scenarios (Current / Soft Landing / Rate Spike / Mild Recession / Severe Crisis / Stagflation) each with calibrated regime probability distributions (Ang & Timmermann 2012).
- Per-scenario shows: regime distribution pips, ensemble weight bars vs current baseline delta, Kelly capital deployment vs baseline.
- Pure client-side via `MetaEnsemble.blendWeights` + `KellyCalculator`.
- **Files:** `frontend/src/components/Dashboard/ScenarioPanel.jsx`, `frontend/src/components/Dashboard/ForecastPanel.jsx`

### Feature: Forecast architecture dropdown + status dots + weight strip (`823457e`)
- `ForecastArchitecture.jsx`: collapsible 8-layer pipeline diagram (data ingestion → 6 models → regime → meta-ensemble → outputs). Defaults collapsed.
- `ForecastComposite`: per-method status dots (glowing when done, dimmed waiting, red error), ensemble weight color bar.
- **Files:** `frontend/src/components/Dashboard/ForecastArchitecture.jsx`, `frontend/src/components/Dashboard/ForecastComposite.jsx`, `frontend/src/components/Dashboard/ForecastPanel.jsx`

### Fix: Vercel build — `f*/2` in JSDoc closed block comment (`9ccfea6`)
- `*/` inside a `/** ... */` block comment terminates the comment, breaking Rollup parse.
- Fixed `KellyCalculator.js` line 10: `Half-Kelly (f*/2)` → `Half-Kelly (f* divided by 2)`.

---

## 2026-04-19 — Phase 5–7 + build fixes (session 1)

**Commits:** `96654e8`, `9db545d`, `9ebf4b6`, `a7264bf`

### Phase 7: EDGAR 10-K/10-Q filing sentiment (`96654e8`)
- `backend/app/services/edgar_filing_provider.py`: fetches CIK from `company_tickers.json`, uses EDGAR EFTS full-text search for highlighted risk-factor / forward-looking excerpts. Fallback: submissions API + 300KB-capped primary document + regex Item 1A/7 extraction. Top-5 holdings by |weight|, ETFs skipped, 24h cache.
- `backend/app/models/forecast.py`: added `edgar_context: Optional[dict]` to `ForecastResponse`.
- `backend/app/services/forecast_engine.py`: fetches EDGAR context, attaches to phase 1 response.
- `frontend/src/hooks/useForecast.js`: expose `edgarContext` state from `p1Data.edgar_context`.
- `frontend/src/components/Dashboard/SentimentPanel.jsx`: rewritten with two tabs — News Headlines (GDELT/FinBERT existing flow) and SEC Filings (EDGAR excerpts, sentence-split before FinBERT, filing type/date/company cards).

### Phase 6: Kelly position sizing (`9db545d`)
- `frontend/src/ml/KellyCalculator.js`: `f* = μ/σ²` from ensemble 90% CI; `σ` from `(p95−p5)/(2×1.645)`; regime confidence multiplier (crisis 0.50, bear 0.70, bull_high_vol 0.85, bull_low_vol 1.00); half/full Kelly toggle; clipped to [0, 2].
- `frontend/src/components/Dashboard/KellyPanel.jsx`: SVG arc gauge, deployment bar, stat grid.

### Fix: Vercel build — `onnxruntime-web` WASM dynamic import (`9ebf4b6`)
- Vite 6 Rollup cannot bundle WASM from dynamic imports. Fix: load from CDN in `index.html` as `window.ort`; externalize in `vite.config.js`; `XGBoostInferer.js` and `MetaEnsemble.js` use `window.ort` instead of `await import()`.

### Phase 5: Regime-conditional meta-learner (`a7264bf`)
- See entry below for full detail.

---

## 2026-04-19 — Phase 5: Regime-Conditional Meta-Learner (Stacked Generalization)

**What:** Ensemble layer on top of the 6 base forecast models using regime-conditional weights.

**New backend files:**
- `backend/app/services/meta_learner.py` — `compute_regime_probs()` (4-state: 2-state HMM × VIX threshold), `get_ensemble_weights()` (Ang & Timmermann 2012 priors), `compute_disagreement()` (Krogh & Vedelsby 1995). Rule-based weights: bull_low_vol→Factor leads; bull_high_vol→XGBoost leads; bear→HMM leads; crisis→GP leads.
- `backend/app/routers/regime.py` — POST /api/regime/current: runs HMM + VIX on equity curve, returns 4-state regime probs + ensemble weights.
- `backend/scripts/train_meta_learner.py` — Offline training script: surrogate base model predictions → per-regime ElasticNetCV (purged walk-forward CV, embargo=21d) → ONNX export to `frontend/public/models/meta/{regime}.onnx`.

**Updated backend files:**
- `main.py` — registered regime router.
- `forecast.py` model — added `regime_probs`, `ensemble_weights` fields to ForecastResponse.
- `forecast_engine.py` — after HMM runs: calls `compute_regime_probs()` + `get_ensemble_weights()`, stores in `_hmm_out_extras`, returns at response level.

**New frontend files:**
- `frontend/src/ml/MetaEnsemble.js` — `computeDisagreement()`, `blendWeights()`, `blendBands()`, `computeEnsemble()`. ONNX per-regime models loaded if available (fallback to rule-based). Disagreement widens outer bands (high→+25% σ, med→+10% σ).
- `frontend/src/components/Dashboard/EnsembleCard.jsx` — featured card above composite chart: ensemble fan chart (violet #c084fc), regime donut (SVG arc, 4 colors), disagreement badge, method weight bars. Cites Wolpert (1992), Ang & Timmermann (2012), Krogh & Vedelsby (1995), Lakshminarayanan et al. (2017).

**Updated frontend files:**
- `ForecastPanel.jsx` — imports EnsembleCard + MetaEnsemble; `useEffect` recomputes ensemble async when results/regime_probs change; EnsembleCard rendered above ForecastComposite.

**Regime classification logic:**
- bull_low_vol: HMM bull_prob fraction where vix_rank ≤ 0.60
- bull_high_vol: remaining bull_prob fraction where vix_rank > 0.60 (linear ramp to 1.0 at 1.0)
- bear: HMM bear_prob fraction where vix_rank ≤ 0.80
- crisis: remaining bear_prob fraction where vix_rank > 0.80 (linear ramp)

---

## 2026-04-18 — Phase 4: Tier 2 Data Providers + Client-Side FinBERT

**What:** Added GDELT news, Reddit mention velocity, Google Trends, and browser-side FinBERT sentiment scoring.

**New backend files:**
- `backend/app/services/news_provider.py` — GDELT 2.0 DOC API, no packages, 12h cache. Returns per-ticker headlines for FinBERT.
- `backend/app/services/reddit_provider.py` — Reddit public JSON API (no auth), 3h cache. Returns 7-day mention count + avg score.
- `backend/app/services/trends_provider.py` — pytrends Google Trends, 24h cache. Returns 7-day z-score vs 90-day baseline.

**New frontend files:**
- `frontend/src/ml/SentimentInferer.js` — loadFinBERT, scoreHeadlines, aggregateScores, deviceCanRunFinBERT. Uses @xenova/transformers v2 (Xenova/finbert, ~80MB, cached in IndexedDB).
- `frontend/src/components/Dashboard/SentimentPanel.jsx` — idle→can_run→downloading→scoring→done state machine. Shows per-ticker sentiment bars with net score ∈ [-1,+1].

**Updated backend files:**
- `forecast.py` model — added `news_context`, `tier2_context` fields to ForecastResponse.
- `forecast_engine.py` — Tier 2 fetch after Tier 1; `_portfolio_reddit()` + `_portfolio_trends()` aggregate helpers; reddit/trends injected into XGBoost, HMM, LSTM metadata; news_context at response level.
- `requirements.txt` — added pytrends>=4.9.2.

**Updated frontend files:**
- `vite.config.js` — exclude @xenova/transformers from pre-bundling (uses dynamic WASM/Workers).
- `package.json` — added @xenova/transformers@^2.17.2.
- `useForecast.js` — extracts news_context from phase1 response, exposes as `newsContext`.
- `ForecastPanel.jsx` — renders SentimentPanel below method cards grid.
- `ForecastMethodCard.jsx` — XGBoost meta: WSB mentions + trends z-score + article count; HMM meta: trends z-score + article count; LSTM meta: trends z-score + article count.

**Tier 2 routing:**
| Source | XGBoost | HMM | LSTM | N-BEATS | Factor | GP |
|--------|---------|-----|------|---------|--------|----|
| FinBERT (browser) | ✓ SentimentPanel | ✓ SentimentPanel | ✓ SentimentPanel | ✓ SentimentPanel | — | — |
| Reddit | ✓ metadata | — | — | — | — | — |
| Google Trends | ✓ metadata | ✓ metadata | ✓ metadata | — | — | — |
| Article count | ✓ metadata | ✓ metadata | ✓ metadata | — | — | — |

---

## 2026-04-18 — Phase 3E: Route Tier 1 Data to HMM, Factor, GP, N-BEATS

**What:** Macro/VIX context now flows to all forecast methods, not just XGBoost.

**Changes to `forecast_engine.py`:**
- `run_all_forecasts()`: fetches `get_macro_features()` + `get_vix_features()` once (merged into `macro_ctx` dict), before the existing insider fetch.
- `forecast_hmm()`: new `macro_context` param → annotates metadata with `macro_env` (expansionary/neutral/restrictive), `yield_curve_regime` (inverted/flat/normal/steep), `vix_regime` (calm/above_avg/elevated).
- `forecast_factor()`: new `macro_context` param → computes `cycle_scale` multiplier on `mkt_rf` premium only (yield curve pct_rank < 0.15 → ×0.70; credit pct_rank > 0.85 → ×0.75). Stored in `macro_adjustment` metadata.
- GP (`var`): macro VIX rank, term slope, and yield curve added to metadata.
- N-BEATS: `macro_context` dict passed through in metadata for client-side display.

**Changes to `ForecastMethodCard.jsx`:**
- HMM strip: shows `macro` (env), `YC rgm`, `VIX rgm` with warns.
- Factor strip: shows `mkt scale` (cycle_scale%) and `10Y-2Y` when macro_adjustment present.
- GP strip: shows `VIX rank`, `VIX slope`, `10Y-2Y`.
- N-BEATS strip: shows `VIX rank`, `10Y-2Y`.

**Academic basis for cycle_scale:**
- Campbell & Cochrane (1999): equity risk premium is counter-cyclical.
- Ludvigson & Ng (2009): macro factors predict equity risk premia.

---

## 2026-04-18 — Phase 3: Tier 1 Data Provider Layer (FRED + VIX → XGBoost 14 features)

**What:** Added live macro/VIX data providers and expanded XGBoost from 9 to 14 features.

**New files:**
- `backend/app/services/fred_provider.py` — Fetches T10Y2Y, BAA10Y, DFII10, DFF from FRED via pandas_datareader. Returns scalar values + rolling percentile ranks. 24h in-process cache. Falls back to long-run neutral averages on network failure.
- `backend/app/services/vix_provider.py` — Fetches ^VIX + ^VIX3M from yfinance. Returns vix_spot, vix_pct_rank, vix_term_slope, vix_contango. Same caching/fallback pattern.

**Updated files:**
- `backend/scripts/train_xgboost.py` — 9→14 features; `download_macro_data()` joins FRED+VIX history; `compute_features()` accepts optional `macro_vals: np.ndarray`; `build_dataset()` accepts optional `macro_df`; meta.json now includes `feature_means`, `macro_features`, `n_macro_features` for inference-time fallback.
- `backend/app/services/forecast_engine.py` — `prepare_xgboost_features()` now calls `get_macro_features()` + `get_vix_features()`; appends 5 macro values to features list; populates `macro_context` dict for the UI card.
- `frontend/src/components/Dashboard/ForecastMethodCard.jsx` — XGBoost meta strip now shows VIX level (warned if pct_rank > 0.75), 10Y-2Y yield curve (warned if inverted), and "macro: neutral" warn badge if FRED/VIX were unavailable.

**Macro features added (5):**
  yield_curve_10y2y, credit_spread_baa, vix_pct_rank, vix_term_slope, real_yield_10y

**Action required:** User must retrain XGBoost locally:
  `cd backend && python scripts/train_xgboost.py`
  Then commit new ONNX files + meta.json.

**Pending:** Phase 3D (SEC/EDGAR insider data), Phase 3E (route macro to HMM + GP + Factor), Phase 4 (Tier 2), Phase 5 (regime-conditional meta-learner).

---

## 2026-04-18 — Phase 2D: Replace VAR → Gaussian Process Autoregression

**What:** Replaced VAR (Vector Autoregression) with a Gaussian Process autoregression (GPAR). VAR required fetching individual asset prices, imposed stationarity + normality assumptions, and used Monte Carlo simulation. GP replaces all of this with exact Bayesian inference on the portfolio return series directly.

**Architecture:** ARD Matérn ν=5/2 kernel (7 lag-specific length scales) + WhiteKernel for noise floor. Rolling 7-day return window as input features. Up to 300 training points (O(n³) tractability cap). Fan chart from Gaussian approximation: cumulative return at T ~ N(Σμₜ, Σσₜ²) — no Monte Carlo needed.

**OOS diagnostics:** Chronological 80/20 split. Predictive R² (skill) + NLPD (negative log predictive density, calibration quality — lower is better).

**Files modified:**
- `backend/app/services/forecast_engine.py` — removed `forecast_var()` and VAR asset-price fetch block; added `forecast_gp(returns, horizon, last_date)`; METHOD_LABELS["var"] → "Gaussian Process (GP)"
- `frontend/src/components/Dashboard/ForecastMethodCard.jsx` — GP citations (Rasmussen & Williams 2006, Roberts et al. 2013, Matérn 1960); COMPLEXITY HIGH; meta strip shows OOS R², NLPD, lookback, kernel
- `frontend/src/components/Dashboard/ForecastPanel.jsx` — LoadingCard label updated; ETA reduced to 15s; removed unused `React`, `BROWSER_METHODS`, `timing` vars

**Status:** COMPLETE. No training required — GP fits at inference time.

---

## 2026-04-18 — Phase 2C: Extend Factor Model → FF5 + Momentum + Quality

**What:** Extended the Factor Model forecast from FF5 to FF5 + Momentum (UMD, Carhart 1997). Quality is already captured by RMW (Novy-Marx 2013 establishes gross profitability = quality). The upgrade adds Momentum beta to the expected-return formula: `μ = RF + Σ(βᵢ × premia) + α`, where Momentum premium = 3.8% annualized.

**Where:** Momentum factor is fetched from Ken French's daily data library (`F-F_Momentum_Factor_daily` via pandas-datareader), which runs during the backtest's FF5 decomposition step. Falls back gracefully to 5-factor if data unavailable.

**Mechanics:** `factor_decomposition.py` now runs a 6-factor OLS regression (FF5 + UMD) when Ken French data is available. The result dict includes `mom`, `mom_t_stat`, `mom_stars`, and `factors_used`. `forecast_factor()` iterates over all keys in `FACTOR_PREMIA` instead of hardcoding 5 factors — making future factor additions a one-liner.

**Files modified:**
- `backend/app/services/factor_decomposition.py` — `_via_ken_french()` fetches `F-F_Momentum_Factor_daily`; `_build_result()` refactored to accept dynamic `factor_keys` list
- `backend/app/services/forecast_engine.py` — `FACTOR_PREMIA` gets `"mom": 0.038`; `forecast_factor()` now sums all premia generically; metadata includes `n_factors`, `mom_beta`, `rmw_beta`; `METHOD_LABELS["factor"]` → "Factor Model (FF5+Mom)"
- `frontend/src/components/Dashboard/FamaFrenchFactors.jsx` — conditionally renders Momentum (UMD) row when `ff5.mom != null`; RMW relabeled "Quality / Profitability (RMW)"
- `frontend/src/components/Dashboard/ForecastMethodCard.jsx` — updated citations (Carhart 1997, Novy-Marx 2013); factor meta strip shows "FF5+Mom"/"FF5" model label, Mom β, Quality β

**Status:** COMPLETE. No training required — uses existing Ken French data download.

---

## 2026-04-18 — Phase 2B: Replace GARCH(1,1) → N-BEATS Neural (Pure-JS Client-Side)

**What:** Replaced GARCH(1,1) with N-BEATS (Neural Basis Expansion Analysis, Oreshkin et al. 2020 ICLR). N-BEATS provides true multi-horizon return forecasts at 5 quantiles via stacked residual MLP blocks. Runs in browser via pure-JS matrix math — no ONNX or TF.js runtime needed (PyTorch 2.4+ ONNX export is broken due to pybind11 signature inspection failure; raw float32 binary export is cleaner and has no dependency).

**Architecture:** 3 residual N-BEATS blocks. Each block: FC(256)×4 → backcast head (30 outputs) + forecast head (21×5 outputs). Backcast subtracted from residual; forecasts summed. Trained with pinball loss (Koenker & Bassett 1978). OOS results: R²=0.345, IC=0.589, near-ideal coverage calibration.

**Browser inference:** `NBeatsInferer.js` loads `weights.bin` (raw float32 binary, ~1-2 MB) + `meta.json` (normalisation params + weight manifest). Implements `linear(x,W,b)` + `relu` in ~50 lines; no runtime dependency. Runs 12 recursive 21-day periods for the 252-day fan chart.

**Export format:** `train_nbeats.py:export_weights()` writes all `model.named_parameters()` as a flat float32 binary with a manifest list `[{name, shape, offset, size}]` stored in `meta.json["weights_manifest"]`.

**Files created/finalised:**
- `backend/scripts/train_nbeats.py` — PyTorch training + pure float32 binary export (no ONNX)
- `frontend/src/ml/NBeatsInferer.js` — pure-JS forward pass loading weights.bin
- `frontend/public/models/nbeats/weights.bin` + `meta.json` — committed model artifacts

**Files modified:**
- `backend/app/services/forecast_engine.py` — `prepare_nbeats_features()` returns last-30-day window
- `frontend/src/hooks/useForecast.js` — XGBoost (ONNX) + N-BEATS (pure-JS) run in parallel via `Promise.all`
- `frontend/src/components/Dashboard/ForecastMethodCard.jsx` — nbeats browser card: "12-period recursive · pure-JS weights"
- `frontend/src/components/Dashboard/ForecastPanel.jsx` — ETA bar label updated

**Status:** COMPLETE. Phase 2B shipped.

---

## 2026-04-18 — Phase 2A: Replace Monte Carlo GBM → XGBoost Quantile (Client-Side ONNX)

**What:** Replaced Monte Carlo GBM with XGBoost gradient-boosted quantile regressors. GBM assumes a random walk with constant μ/σ — no learned signal. XGBoost learns nonlinear interactions between 9 market microstructure features (momentum at multiple horizons, realized vol, vol regime, RSI). Per Gu, Kelly & Xiu (2020, RFS), XGBoost dominates parametric models on OOS stock return prediction.

**Architecture:** XGBoost runs CLIENT-SIDE via ONNX Runtime Web (same pattern as Attention-LSTM via TF.js). Server does feature engineering only; browser loads 5 ONNX models (one per quantile: p5/p25/p50/p75/p95). Fan chart extrapolated from 21-day predictions to 252 days via median compounding + √(t/21) spread scaling.

**Training:** sklearn `GradientBoostingRegressor(loss='quantile')` Pipeline with StandardScaler → skl2onnx ONNX export. Pipeline includes scaler as preprocessing nodes so browser passes raw features. Purged walk-forward CV with 21-day embargo (López de Prado 2018). Expanding window, 252-day minimum train, 20% held-out test.

**Files created:**
- `backend/scripts/train_xgboost.py` — training script. Downloads 15-asset universe (2010-2024), builds 9-feature dataset (~90K+ samples), does walk-forward OOS eval, trains 5 quantile models on full 80% data, exports to ONNX.
- `frontend/src/ml/XGBoostInferer.js` — ONNX Runtime Web inference. Loads 5 ONNX sessions lazily (cached). Extrapolates 21-day quantile predictions to 252-day fan chart.

**Files modified:**
- `backend/app/services/forecast_engine.py` — removed `forecast_monte_carlo()`, added `prepare_xgboost_features()` (returns 9 raw features + display metadata), updated METHOD_COLORS/METHOD_LABELS/dispatch
- `frontend/src/hooks/useForecast.js` — Phase 1B client inference after Phase 1 server response. PHASE1_METHODS now `['xgboost', 'garch', 'factor']`. Loading state gains `xgb` field.
- `frontend/src/components/Dashboard/ForecastMethodCard.jsx` — xgboost citations (Chen/Guestrin 2016, Gu/Kelly/Xiu 2020, Friedman 2001), complexity badge "MED", MetaStrip shows OOS R², vol regime, RSI-14, n_obs
- `frontend/src/components/Dashboard/ForecastPanel.jsx` — xgboost in METHOD_ORDER/PHASE1_METHODS, XGB_EST_MS=3000, EtaBar handles 4 phases (p1/xgb/p2/lstm), browser loading card for xgboost
- `frontend/package.json` — added `onnxruntime-web ^1.20.1`

**Pending (user action required):**
- Run `cd backend && pip install scikit-learn skl2onnx onnx yfinance && python scripts/train_xgboost.py`
- Commit `frontend/public/models/xgboost/*.onnx` files
- Deploy + test forecast tab

**Phase 2A approval gate:** After training, verify OOS R² > 0 (any positive predictive signal beats GBM) and fan chart looks sensible on a test portfolio.

---

## 2026-04-17 — Phase 1: Move Attention-LSTM to Browser (TF.js) — RAM reclamation

**What:** Moved Attention-LSTM inference from the Railway server to the user's browser via TensorFlow.js. This eliminates `tensorflow-cpu` (~450MB) from the server, dropping steady-state RAM from ~680MB (over free-tier limit) to ~230MB (well within 512MB).

**Architecture change:**
- **Before:** Server trains + runs LSTM at request time (TF imports 450MB on first call, stays resident)
- **After:** Server engineers features + returns a scaled 60-day window; browser loads pre-trained TF.js model and runs 200 MC Dropout passes locally

**Three-phase fetch pattern:**
1. Phase 1 (server, ~1-3s): Monte Carlo, GARCH, Factor — unchanged
2. Phase 2 (server, ~5-15s): HMM, VAR + LSTM feature window (no TF on server)
3. Phase 3 (browser, ~2-5s): Attention-LSTM 200 MC Dropout passes via TF.js

**Files modified:**
- `backend/app/services/forecast_engine.py` — replaced `forecast_lstm()` + `_build_attention_lstm()` + `_LSTM_CACHE` with `prepare_lstm_features()` (feature engineering + scaler only). Removed `hashlib` import.
- `backend/requirements.txt` — removed `tensorflow-cpu>=2.16`. Added comment explaining rationale.
- `frontend/src/ml/LSTMInferer.js` — NEW. Loads `/models/lstm/model.json` via `tf.loadLayersModel()`. Runs vectorised MC Dropout: all 200 passes batched at each of 252 horizon steps (252 model.apply() calls, not 200×252). Returns p5/p25/p50/p75/p95 bands as cumulative % returns.
- `frontend/src/hooks/useForecast.js` — added Phase 3 step: after Phase 2 server response, dynamically imports `LSTMInferer.js` and runs client-side inference. Added `loading.lstm` state. `mergeResults` skips lstm until client inference completes.
- `frontend/src/components/Dashboard/ForecastPanel.jsx` — `EtaBar` handles three phases (p1/p2/lstm). LSTM loading card shows "Computing in browser / TensorFlow.js" instead of generic skeleton. Phase 2 ETA revised to 20s (was 75s — LSTM removed from server).
- `frontend/src/components/Dashboard/ForecastMethodCard.jsx` — `browserCompute` prop for LSTM card. `getMetaItems` for lstm shows `engine: TF.js browser`, `passes: 200`, `attention: Bahdanau`.
- `frontend/package.json` — added `@tensorflow/tfjs ^4.22.0`

**Files created:**
- `backend/scripts/train_lstm.py` — local training script. Downloads 15-asset universe (2010-2024), trains Attention-LSTM(64)+Bahdanau attention on chronological 70/15/15 split with early stopping. Exports to `frontend/public/models/lstm/` via tensorflowjs_converter.
- `frontend/public/models/lstm/README.md` — setup instructions for generating model files.

**RAM baseline endpoint:** `GET /api/debug/memory` (added in Phase 0, commit 86984da). Call before + after deploy to verify RAM drop.

**Decisions:**
- Vectorised MC Dropout inference: stack all 200 windows into a batch at each step → 252 model.apply() calls instead of 50,400. Runs in ~2-5s in modern browsers.
- Dynamic import of LSTMInferer.js: TF.js (~3MB gzipped) only loads when LSTM runs, not on app start.
- Model served from Vercel static assets (frontend/public/), zero Railway cost.
- Training script trains a generic model on diverse assets; per-portfolio MinMaxScaler normalisation makes features comparable across portfolios.

**Pending (user action required):**
- Run `cd backend && pip install tensorflow tensorflowjs && python scripts/train_lstm.py` to generate the model files
- Commit `frontend/public/models/lstm/model.json` + `.bin` files
- Deploy to Railway → call `GET /api/debug/memory` to verify RAM drop to <250MB

**RAM target after this phase:** ~230MB steady-state (no TF import)

---

## 2026-04-17 — Forecast: actual dollar values, Attention-LSTM, ETA progress bar

**What:** Three improvements to the Forecast tab:

1. **Actual portfolio values chart** — ForecastComposite and ForecastMethodCard now show real dollar portfolio values (e.g. $10,000 → $17,400) instead of rebased % returns. Forecast lines are projected as `last_value × (1 + p_pct/100)` anchored to the final historical equity value. Y-axis formatted as `$10k`, `$1.2M` etc.

2. **Attention-LSTM** (CS230 Stanford 2020, Bahdanau 2015) — replaced stacked LSTM(64→32) with a single LSTM(64) encoder + Bahdanau temporal attention mechanism. Attention learns which past hidden states are most relevant (e.g. recent volatility regimes vs. distant history). `tensorflow-cpu` enabled in requirements.txt. New architecture: `Attention-LSTM(64) → Dense(32) → Dense(1)`.

3. **ETA progress bar** — `EtaBar` component in ForecastPanel uses `setInterval(250ms)` to show live elapsed time and estimated remaining time. Phase 1 estimated at 4s, Phase 2 (HMM+VAR+Attention-LSTM) at 75s. Bar fills smoothly; never completes until the fetch resolves.

**Files modified:**
- `backend/requirements.txt` — uncommented tensorflow-cpu
- `backend/app/services/forecast_engine.py` — `_build_attention_lstm()` helper + updated `forecast_lstm()`
- `frontend/src/hooks/useForecast.js` — added `timing` state and `p1StartRef`/`p2StartRef` refs
- `frontend/src/components/Dashboard/ForecastComposite.jsx` — actual dollar values
- `frontend/src/components/Dashboard/ForecastMethodCard.jsx` — `lastValue` prop, dollar Y-axis, Attention-LSTM citations
- `frontend/src/components/Dashboard/ForecastPanel.jsx` — EtaBar, pass `lastValue`, updated copy

---

## 2026-04-16 — Fix SplitView crash: remove useEffect hook rule violation

**What:** `SplitView.jsx` crashed at runtime due to a React hook called conditionally inside a render block.

**Root cause:** A `useEffect(() => { if (activeBacktest) setRightTab('results') }, [activeBacktest])` was placed after a conditional computation that used `let` assignments — mixing hook calls with conditional logic violates Rules of Hooks.

**Fix:** Removed the `useEffect` entirely. The `rightTab` persists across backtests (user keeps their tab selection), which is acceptable UX. Refactored right-panel visibility from `showRightTabs` boolean to two explicit `isScreenerView` / `isPortfolioView` derivations, and replaced the ternary chain with parallel `{condition && <JSX/>}` blocks for clarity.

**Files modified:** `frontend/src/components/Layout/SplitView.jsx`

---

## 2026-04-16 — Predictive Analytics: 6-Method Forecast Tab

**What:** Added a new "Forecast" tab to the right results panel. Clicking it runs 6 research-backed probabilistic forecasting methods on the active portfolio, producing 12-month fan-band charts (p5/p25/p50/p75/p95) for each method.

**Methods implemented:**

1. **Monte Carlo (GBM)** — Black & Scholes (1973), Merton (1969). Constant drift/vol GBM, vectorised over 1,000 paths. Complexity: LOW.
2. **GARCH(1,1)** — Engle (1982), Bollerslev (1986). Time-varying conditional volatility with bootstrapped fat-tail residuals. Ljung-Box OOS diagnostic. Complexity: MED.
3. **Hidden Markov Model** — Hamilton (1989), Ang & Bekaert (2002). 2-state Bull/Bear HMM via Baum-Welch EM with 10 random restarts. Returns current regime probability and regime-conditional simulation. Complexity: MED.
4. **Fama-French Factor Forecast** — Fama & French (2015), Cochrane (2011), Damodaran (2024). Replaces naive historical mean with factor-anchored expected return from FF5 loadings × consensus premia. Idiosyncratic vol as noise term. Complexity: LOW.
5. **VAR Multi-Asset** — Sims (1980), Campbell et al. (2003). Vector autoregression on individual asset returns, portfolio reconstructed by weights. AIC lag selection; OOS residual covariance used for simulation. Granger-causality test badge. Complexity: MED.
6. **LSTM Neural Net** — Fischer & Krauss (2018), Hochreiter & Schmidhuber (1997), Gal & Ghahramani (2016). LSTM(64)→LSTM(32)→Dense(1) with MC Dropout for Bayesian uncertainty bands. Complexity: HIGH.

**Strict out-of-sample / anti-overfitting measures (Lopez de Prado, 2018; Bailey & Lopez de Prado, 2014):**
- Walk-forward 80/20 train/test split on all parametric models (GARCH, HMM, VAR)
- LSTM uses chronological 70/15/15 split with early stopping on validation loss (patience=10)
- LSTM MC Dropout (training=True at inference) produces 200 stochastic passes for CI bands
- No random k-fold anywhere — temporal ordering always preserved
- OOS diagnostics returned per method and displayed in the UI (Ljung-Box, OOS R², OOS MSE)
- Factor model uses 60+ year Ken French consensus premia instead of in-sample historical mean

**Citation format:** Matched exactly to `FamaFrenchFactors.jsx` — `📄` footer on every card, `?` InfoTooltip with description + citations + OOS methodology note.

**UI architecture:**
- Two-phase fetch: Phase 1 (MC, GARCH, Factor ~1–2s) renders immediately; Phase 2 (HMM, VAR, LSTM ~10–40s) fills in after
- Full-width composite chart: all 6 median lines overlaid with historical equity curve + vertical forecast-start marker
- 2×3 card grid: each card has fan chart, metadata strip, citation footer, complexity badge, compute time
- Forecast tab only shown when a portfolio backtest is active; resets to Results tab on new backtest

**New backend files:**
- `backend/app/models/forecast.py` — Pydantic request/response schemas
- `backend/app/services/forecast_engine.py` — all 6 models with docstring references
- `backend/app/routers/forecast.py` — `POST /api/forecast` endpoint

**New frontend files:**
- `frontend/src/hooks/useForecast.js` — two-phase fetch hook
- `frontend/src/components/Dashboard/ForecastPanel.jsx` — tab container
- `frontend/src/components/Dashboard/ForecastMethodCard.jsx` — fan chart card with citations
- `frontend/src/components/Dashboard/ForecastComposite.jsx` — composite overlay chart

**Modified files:**
- `backend/app/main.py` — added forecast router
- `backend/requirements.txt` — added arch, hmmlearn, scikit-learn, statsmodels (tensorflow-cpu commented out pending RAM check)
- `frontend/src/components/Layout/SplitView.jsx` — Results/Forecast tab bar on right panel

**Note on LSTM deployment:** `tensorflow-cpu` is commented out in `requirements.txt`. Railway Starter Plan has ~512MB RAM; TF CPU import footprint is ~450MB. Uncomment and upgrade to Pro plan ($5/mo, 8GB RAM) before enabling LSTM in production. All other 5 methods work without TF.

---

## 2026-04-16 — System prompt token optimisation

**What:** Trimmed both AI system prompts by ~63% to reduce token spend on every cache-miss call.

**UNIFIED_SYSTEM_PROMPT** (185 lines → 67 lines, ~1,200 tokens saved):
- Removed all 3-example routing blocks per route
- Metric guide: 5 verbose lines → 1 compact pipe-separated line
- Universe guide: reformatted to dense no-quotes layout
- Date range section: removed redundant enforcement examples (code handles this)
- display_config.narrative template: 10-line code block → 1-line inline description
- Backtest Integrity: 5 bullets → 2 compact lines

**SCREENER_SYSTEM_PROMPT** in `screener_ai.py` (48 lines → 18 lines, ~600 tokens saved):
- Same compression pattern — all key parameters now on single pipe-separated lines

Both prompts retain `cache_control: {"type": "ephemeral"}` so savings apply on every cache miss (first call + any call after 5-minute TTL expiry).

**Files modified:** `backend/app/services/unified_ai.py`, `backend/app/services/screener_ai.py`

---

## 2026-04-13 — Fix long/short backtest engine: position-dollar tracking

**What:** Dollar-neutral long/short portfolios (e.g. NVDA +1 / AMD -1) produced absurd results — weight drift chart showed ±6,000,000%, and the equity curve was also wrong.

**Root cause:** The weight drift update in `run_backtest` normalised by `weight_sum = Σ(new_weights)`. For a dollar-neutral pair, `weight_sum ≈ 0` (longs and shorts cancel). Dividing by near-zero inflated weights to ±100x. Those exploded weights were then used for the *next day's return calculation* (`daily_ret = current_weights · day_rets`), so e.g. weights of 102 and -101 were applied to 1% returns giving 1% instead of ~0%, vastly overstating portfolio gains and compounding the error every day.

**Fix:** Replaced the normalise-by-weight-sum approach with explicit position-dollar tracking:
- `position_dollars[t]` = dollar value of each position (signed: negative = short)
- `daily_pnl = Σ(position_dollars[t] × ret[t])` → correct portfolio P&L regardless of net weight
- `portfolio_value` tracked independently as running sum of P&L
- Drifted weights for the chart = `position_dollars[t] / portfolio_value` → stays in ±1–2x range for pair trades

Verified correct for both long-only (60/40, weights still sum to 1 after drift) and dollar-neutral pairs (weights stay in the ±1x range expected for a pair trade).

**Files modified:** `backend/app/services/backtest_engine.py` — `run_backtest()` function

---

## 2026-04-13 — Short positions fix + warm universe cache + manual builder UX

**What:** Three improvements in one commit.

**1. Fix: AI never producing negative weights for long/short requests**
Root cause: the unified AI system prompt acknowledged that weights *can* be negative but gave no guidance on *when* to use them. Claude defaulted to all-positive weights even when the user asked for "long short pairs", "pair trade", "market neutral", etc.
Fix: added an explicit `### Long/Short & Market-Neutral Portfolios` section to `UNIFIED_SYSTEM_PROMPT` in `unified_ai.py` that covers:
- When to trigger (signal phrases: "long short", "pair trade", "market neutral", "long X short Y", "hedge with")
- Exact negative weight requirement with examples (e.g. long AAPL +1.0, short QQQ -1.0)
- Dollar-neutral, 130/30, and market-neutral construction patterns
- Recommended `daily` rebalance for pair trades to maintain dollar-neutral exposure

**2. Warm universe pre-cache on startup**
Added a background startup task in `main.py` that pre-fetches 10 years of daily prices for 50 tickers (32 ETFs + 20 high-volume stocks) into the existing SQLite price store.
- Fires via `asyncio.create_task(_warm_price_cache())` inside the `lifespan` context — non-blocking, app serves requests while cache fills.
- Subsequent restarts are nearly free (SQLite L2 hit for all covered tickers/dates).
- Estimated storage: ~8 MB total.
- Screener universe guide in `unified_ai.py` updated to note which tickers are pre-cached, so Claude prefers them for fast response.

**3. Manual builder: visual short position indicators**
- Weight column header now shows `(−=short)` hint so users know negative values work.
- Rows with negative weight render with a red border and red weight text, making shorts visually distinct from longs.

**Files modified:**
- `backend/app/services/unified_ai.py` — long/short system prompt section + updated screener universe guide
- `backend/app/main.py` — added `WARM_UNIVERSE` list + `_warm_price_cache()` background task
- `frontend/src/components/Chat/ManualBuilderPanel.jsx` — weight header hint + short row styling

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

## 2026-04-13 — Merge AI Portfolio Builder + AI Stock Screener into single unified chat

**Commit:** `2f2d4b1`

**What:** Replaced the two separate chat panels (ChatPanel for portfolio, StockScreenerPanel for screener) and the three-tab layout with a single unified chat panel. Claude now routes to the right pipeline based on intent keywords in the user's message. This was the largest structural change to the frontend.

**Backend — new files/changes:**
- `backend/app/services/unified_ai.py` (new, 433 lines) — single AI service with all three tools: `get_asset_statistics`, `construct_portfolio`, `run_screen`. System prompt has explicit ROUTING GUIDE with signal phrases ("top / best / each quarter" → screener; "portfolio / allocate / backtest" → portfolio). Portfolio path runs full agentic research loop; screener path runs real window screener then feeds results back to Claude for a data-aware narrative.
- `backend/app/routers/unified.py` (new) — `POST /api/unified/chat`. Returns `{type: "portfolio"|"screener"|"clarification", ...result}`.
- `backend/app/main.py` — registered unified router.

**Frontend — changed files:**
- `frontend/src/components/Chat/UnifiedChatPanel.jsx` (new, replaces ChatPanel + StockScreenerPanel) — single chat panel. Empty state shows "Portfolio Builder" (blue) + "Asset Screener" (purple) capability pills with a short "how to use" card. 4 suggestion chips (2 portfolio/blue, 2 screener/purple). Loading indicator adapts label to detected result type.
- `frontend/src/components/Layout/SplitView.jsx` — reduced from 3 tabs to 2 (AI Assistant + Manual Build). Right panel dynamically renders `ScreenerResults`+`RotationPanel` or `ResultsPanel` based on the `type` field of the last result. All import flows (screenerImport, rotationImport) preserved.
- `frontend/src/components/Layout/App.jsx` — simplified; chat state now managed inside SplitView/UnifiedChatPanel. Header updated to "AI Investment Analyst".

**Decisions:**
- "Unified" approach chosen over tabs because screener and portfolio use the same input box and should feel like one conversation
- Screener vs portfolio routing is done by Claude (not regex/client-side) — more robust to paraphrasing
- `type` field in the response drives which right panel is shown: screener gets `ScreenerResults` split-view; portfolio gets `ResultsPanel`; clarification gets only the chat message
- `StockScreenerPanel.jsx` and `ChatPanel.jsx` retained in codebase but no longer used

---

## 2026-04-13 — Fix portfolio builder "Request failed" + optimise agentic loop

**Commit:** `52ab884`

**What:** Two bugs and three performance optimisations in `unified_ai.py` right after the merge.

**Bugs fixed:**
- Wrong backtest function called: was calling `run_backtest(portfolio_input)` (low-level, takes a prices DataFrame) instead of `run_full_backtest(portfolio_input)` (the high-level wrapper that fetches prices, runs backtest, computes all metrics, rolling metrics, correlation matrix, FX curves). This caused every portfolio request to throw a TypeError.
- Missing `apply_strategy()` call for non-custom strategies (min_variance, max_sharpe, risk_parity, etc.) — weights were being sent unoptimised. Added, matching the logic in `chat.py`.
- Removed duplicate `import os` at module level.

**Optimisations:**
- `MAX_ITERATIONS` reduced from 6 → 4. Most requests complete in ≤2 turns; 4 gives headroom without runaway loops.
- `max_tokens` for final chat bubble reduced 1024 → 512 (3–5 bullets needs far less).
- System prompt: explicit rule to skip `get_asset_statistics` for portfolios where the user already gave weights or named a known strategy — reduces most common case from 3 API calls to 2.
- System prompt: "limit to one `get_asset_statistics` call" — prevents multi-call research loops.
- Screener narrative `max_tokens` also capped at 512.

**Files modified:** `backend/app/services/unified_ai.py`

---

## 2026-04-13 — Reduce suggestion chips to 4 per AI chat panel

**Commit:** `b380aab`

**What:** Trimmed the suggestion chips in both `ChatPanel` and `StockScreenerPanel` from 6 to 4 each before the unified merge. Reduces visual noise and makes the most relevant examples more prominent.

**Files modified:** `frontend/src/components/Chat/PromptSuggestions.jsx`, `frontend/src/components/Chat/StockScreenerPanel.jsx`

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
