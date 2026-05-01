# Forecast Engine Upgrade — Plan Document

> **Created:** 2026-04-30 · **Status:** PLAN ONLY — not yet implemented · **Owner:** Ben
> **Reference commit (current state):** `54b2b7b`
>
> This is the working blueprint for the next big forecast change. Future Claude sessions: read this end-to-end before touching `forecast_engine.py`, `useForecast.js`, or any of the method-specific files. Each phase has a "Files to touch" subsection — work down those lists rather than re-deriving the surface area.

---

## TL;DR

1. **Drop XGBoost and Factor** from the ensemble. Both were holdovers that didn't earn their keep — XGBoost's 21d cap + crude √t extrapolation, Factor's static FF5 priors that don't adapt to the user's portfolio.
2. **Add Google TimesFM 2.5** as a foundation-model anchor. Zero-shot, transformer-based, takes the portfolio equity curve + 4 research-validated covariates. **Self-hosted on Railway Hobby (always-warm)** — HuggingFace's free Serverless Inference API does not serve the `time-series-forecasting` pipeline (verified 2026-05-01: `{"error":"Model not supported by provider hf-inference"}`). We load `TimesFm2_5ModelForPrediction.from_pretrained(...)` once at process startup and keep weights resident for zero-cold-start inference. Estimated +$10-20/mo on Railway Hobby after credits.
3. **Replace internal 80/20 splits with a shared OOS holdout window** at `[T−60d, T−30d]`. Every method is fit on `[start, T−60d]`, forecasts the 30-day shadow window, and the result is overlaid on the composite chart against actual prices.
4. **Re-weight the ensemble by OOS R²** (clip negatives to 0 → normalize → floor TimesFM at 30%). Drop the `_REGIME_WEIGHTS` rule-based priors entirely along with the Ang & Timmermann (2012) citation.
5. **Frontend cleanup**: remove the EnsembleDegradationStrip, redo the architecture flow, streamline method cards to two columns (IS R², OOS R²).

End state: 5 methods (N-BEATS, HMM, GP, LSTM, TimesFM), one shared evaluation regime, R²-driven ensemble, TimesFM as the heavyweight prior with a 30% floor.

---

## Current state snapshot (commit `54b2b7b`)

```
Phase 1 (server, ~2 s warm)
├── XGBoost  ──(features only)──▶ browser ONNX → 21d quantile, √t extrap to 63d
├── N-BEATS  ──(features only)──▶ browser pure-JS → 3 × 21d recursive
└── Factor   ──(server)─────────▶ FF5 GBM, 63d simulation

Phase 2 (server, ~20 s warm / 60-90 s cold)
├── HMM      ──(server, parallel with providers)──▶ regime-conditional MC sim
├── GP       ──(server, parallel with providers)──▶ Matérn-5/2 ARD AR rollout
└── LSTM     ──(features only)──▶ browser TF.js MC Dropout × 63 steps

Ensemble = REGIME_WEIGHTS[dominant_regime] × per-timestep renormalization
         (Ang & Timmermann 2012 priors — to be removed)
```

---

## Target state

```
Phase 1 (server, ~2 s warm)
└── N-BEATS  ──(features only)──▶ browser pure-JS → 3 × 21d recursive

Phase 2 (server, ~15-25 s warm)
├── HMM         ──(server, parallel with providers)
├── GP          ──(server, parallel with providers)
├── LSTM        ──(features only)──▶ browser TF.js
└── TimesFM 2.5 ──(server, in-process always-warm)──▶ transformers + torch CPU
                              inputs: portfolio equity curve + 4 covariates
                              memory: ~1.5 GB resident (loaded at startup)

For every method, run TWICE:
  (a) truncated fit on [start, T−60] → forecast [T−60, T−30] = "shadow"
      → OOS R² computed against actual prices in the holdout
  (b) full fit on [start, T] → forecast [T, T+63] = "forward"
      → displayed on composite chart

Ensemble weights = OOS R² (clipped at 0) → normalize → TimesFM floor 0.30
```

---

# Phase A — Drop XGBoost and Factor

## A.1 Rationale

- **XGBoost**: trained on a 21-day target, extrapolated to 63d via √t scaling. The √t step is heuristic, not learned. Ensemble weight already collapsed to ~12-25% pre-cap and 0% post-cap. Two providers (insider, sec_provider) feed it metadata that nothing else needs.
- **Factor**: closed-form GBM seeded by static FF5 long-run premia (Damodaran 2024 estimates baked in). Doesn't adapt to the user's specific portfolio mix; just a thinly-disguised constant return assumption.
- Both are pre-foundation-model designs. Once TimesFM is in, they're redundant.

## A.2 Backend deletions

| Path | Action |
|---|---|
| `backend/app/services/factor_decomposition.py` | DELETE — only consumed by `forecast_factor` and one ai_service tool description |
| `backend/scripts/train_xgboost.py` | DELETE |
| `backend/app/services/forecast_engine.py::forecast_factor` | DELETE function (~80 lines) |
| `backend/app/services/forecast_engine.py::prepare_xgboost_features` | DELETE function (~60 lines) |
| `backend/app/services/forecast_engine.py::FF5_PREMIA` constant | DELETE if defined |
| `backend/scripts/test_forecast_methods.py` | Drop xgboost/factor cases from `BUDGETS_S` and run loop |
| `.github/workflows/retrain-xgboost.yml` (if exists) | DELETE |

## A.3 Backend in-place edits

| File | Change |
|---|---|
| `backend/app/services/forecast_engine.py` | `METHOD_LABELS`, `METHOD_COLORS` — drop xgboost+factor entries. Method dispatch loop — drop the `xgboost`/`factor` branches. Update module docstring "6 methods" → "5 methods". |
| `backend/app/models/forecast.py` | `methods: list[str] = [...]` default — drop "xgboost", "factor". Add "timesfm". |
| `backend/app/routers/forecast.py` | Update docstring `Methods: ... | factor | ...` line. |
| `backend/app/services/ai_service.py` | If the AI tool prompt mentions XGBoost/Factor as advisor methods, scrub. |
| `backend/app/services/sec_provider.py` | KEEP — insider data is a TimesFM covariate (see Phase B). Don't delete. |
| `backend/app/services/edgar_filing_provider.py` | KEEP — drives FinBERT sentiment in browser. |

## A.4 Frontend edits (lightweight per user — full polish in a later pass)

| File | Action |
|---|---|
| `frontend/src/ml/XGBoostInferer.js` | DELETE |
| `frontend/public/models/xgboost/` | DELETE entire directory (~512 KB Vercel asset) |
| `frontend/src/hooks/useForecast.js` | Remove `PHASE1_METHODS` xgboost entry. Remove the `xgbResult` Phase 1B block (~50 lines). Remove `HORIZON_CAPS_DAYS.xgboost` (no longer relevant — N-BEATS cap is the only one left and it equals the horizon). Update `xgbStartRef`, `xgbStartMs`, etc. |
| `frontend/src/components/Dashboard/ForecastPanel.jsx` | Remove `XGB_EST_MS`, `xgb` from `activePhase` priority chain, all `xgbStartRef` props. Remove "Phase 1B: XGBoost" label. |
| `frontend/src/components/Dashboard/ForecastComposite.jsx` | `METHOD_ORDER`, `METHOD_LABELS`, `METHOD_COLORS` — drop xgboost+factor; add timesfm. |
| `frontend/src/components/Dashboard/ForecastMethodCard.jsx` | Drop xgboost+factor from `METHOD_DESC`, `METHOD_COLOR_TRANSPARENT`, etc. |
| `frontend/src/components/Dashboard/ForecastSnapshotCards.jsx` | Drop xgboost+factor from method dot row. |
| `frontend/src/components/Dashboard/ForecastExport.jsx` | Drop xgboost+factor from `METHOD_ORDER`/`LABELS`/`COLORS`. Methodology footer text. |
| `frontend/src/ml/MetaEnsemble.js` | Drop xgboost+factor from `REGIME_WEIGHTS` (about to be killed entirely in Phase C anyway). |

## A.5 Verification checklist before merging Phase A

- [ ] `grep -ri "xgboost\|factor" backend/app/ | grep -v "_pycache\|test"` returns only the test_forecast harness + sec_provider's "insider trading" (unrelated). Should be zero forecast-engine references.
- [ ] Frontend bundle still builds: `cd frontend && npm run build`.
- [ ] Forecast endpoint returns successfully with `methods: ["nbeats", "hmm", "var", "lstm"]`.
- [ ] No 404s in browser console for `/models/xgboost/*`.

---

# Phase B — Add Google TimesFM 2.5

## B.1 Why TimesFM 2.5

TimesFM is Google Research's foundation model for time-series forecasting (Das et al., 2024 — *A decoder-only foundation model for time-series forecasting*, ICML 2024). Properties that matter for this app:

- **Zero-shot.** No retraining per portfolio, no GH Actions workflow. Inference only.
- **Pre-trained on 100B+ time-series points** spanning finance, energy, retail, weather. Generalises rather than memorises.
- **Probabilistic output.** Returns quantile bands (p10/p50/p90 by default; can be remapped to p5/p25/p50/p75/p95 to match the rest of the ensemble).
- **Covariate support.** TimesFM-X variant (or 2.5's covariate head) takes exogenous features alongside the target series.
- **3-month horizon is well within its training window.** TimesFM's recommended max horizon is 512 steps; we ask for 63.

**Verified at plan-write time (2026-05-01)** — model is live, Apache 2.0, not gated:
- HF Hub ID: `google/timesfm-2.5-200m-transformers` (the `-pytorch` variant exists but `-transformers` is the canonical, transformers-library-compatible build, 234k+ downloads)
- Class: `TimesFm2_5ModelForPrediction` (now in `transformers` main)
- Params: 231M (FP32 weights ~925 MB on disk)
- Single quantized fork exists (`pdufour/timesfm-2.5-200m-transformers-onnx`, FP32 ONNX, ~2 GB) — no INT8/FP16 published

## B.2 Deployment decision: self-host always-warm on Railway Hobby

### Why not HuggingFace Inference API

**Originally planned to use HF Serverless Inference API. Verified 2026-05-01: this does not work.**

HF's Serverless Inference API does NOT support the `time-series-forecasting` pipeline. Both the legacy `api-inference.huggingface.co/models/...` URL and the new `router.huggingface.co/hf-inference/models/...` router return:

```
{"error":"Model not supported by provider hf-inference"}
```

The `endpoints_compatible` tag on the model card refers to **paid** HF Inference Endpoints (~$0.06/hr CPU, ~$5-15/mo with scale-to-zero), not the free Serverless API.

### Memory math (revised for Railway Hobby)

User upgraded from Railway free tier to **Hobby plan**. Hobby ceilings:
- Up to **8 GB RAM per replica** (vs 512 MB on free tier)
- Up to **5 GB storage** per service
- $5/mo subscription + $5 credit + usage-based billing

TimesFM footprint:
- FP32 transformers weights: ~925 MB on disk, ~1.5 GB peak runtime RAM (model + activations + tokenizer + torch overhead)
- Fits comfortably in 8 GB ceiling

### Recommendation: self-host, always-warm

Load TimesFM at FastAPI startup, keep resident for the lifetime of the process. Trade-off chosen explicitly per user instruction: *"i want performance"*.

| Pattern | Avg RAM | First-request latency | Mo cost on Hobby |
|---|---|---|---|
| Always-warm (CHOSEN) | ~1.8 GB | 2-15s (just inference) | ~$15-25 |
| Lazy + 15min idle-unload | ~600 MB avg | 10-30s on cold load | ~$5-10 |
| Subprocess-per-request | ~300 MB | 10-30s every request | ~$5-10 |

Always-warm wins on UX: every forecast hits a hot model. Cost premium is ~$10-15/mo, well within the Hobby + credit budget for an app that's a personal/portfolio project.

### Implementation pattern

```python
# timesfm_provider.py — module-level lazy singleton, loaded on first import.
# FastAPI's lifespan handler eager-fires the load at startup so the first
# /api/forecast doesn't pay the load cost.

_model: TimesFm2_5ModelForPrediction | None = None
_load_lock = threading.Lock()

def get_model():
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                _model = TimesFm2_5ModelForPrediction.from_pretrained(
                    "google/timesfm-2.5-200m-transformers",
                    torch_dtype=torch.float32,  # FP16 not officially supported
                ).eval()
    return _model
```

In `main.py`:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fire-and-forget warmup — don't block startup if HF Hub is slow
    asyncio.create_task(asyncio.to_thread(get_model))
    yield
```

### Fallback chain

1. **Inference timeout (>30 s)**: skip TimesFM for this request, log warning, ensemble auto-renormalises across remaining methods
2. **Model load failure on startup** (network, OOM): log fatal, mark `_model = "FAILED"` sentinel, every TimesFM call returns `{ error: "model_unavailable" }` — ensemble auto-renormalises
3. **OOM during inference** (very long context): truncate `past_values` to last 1024 points, retry once

### Disk + Docker considerations

- Model weights cached under `~/.cache/huggingface/hub` — ~925 MB. Hobby gives 5 GB → fine
- First container start: downloads weights from HF Hub (~30-60 s). Subsequent starts use cache
- **Bake weights into the Docker image** to skip the first-start download: `RUN python -c "from transformers import TimesFm2_5ModelForPrediction; TimesFm2_5ModelForPrediction.from_pretrained('google/timesfm-2.5-200m-transformers')"` — adds ~925 MB to image size but eliminates cold-deploy weight download

### Auth + secret management

- **No HF token required for inference** (model is public Apache 2.0, downloaded once via Hub API)
- `HF_API_TOKEN` may still be useful for higher Hub download quotas during builds — keep the env var for that, but don't fail startup if missing

## B.3 Covariate stack — 4 research-validated features

TimesFM 2.5's covariate head accepts a `[T, K]` array of exogenous features alongside the `[T]` target series. We already fetch every one of these via existing providers — **no new API integrations required**. All four are causally observable (no look-ahead).

| # | Covariate | Provider | Why predictive (citation) |
|---|---|---|---|
| 1 | **VIX percentile rank (5y)** | `vix_provider.py` | Bekaert & Hoerova (2014, *Journal of Econometrics*) — VIX decomposes into expected vol + variance risk premium; the latter has strong forecasting power for equity returns at 1-3 month horizons. |
| 2 | **Yield-curve term spread (10Y − 2Y)** | `fred_provider.py` | Estrella & Hardouvelis (1991, *J. Finance*) and Ang, Piazzesi & Wei (2006, *J. Econometrics*) — term spread is the most consistent equity premium predictor in the macro literature; flips sign in recessions. |
| 3 | **News sentiment EMA (7d, FinBERT-derived)** | `news_provider.py` (GDELT) → already aggregated portfolio-level | Tetlock (2007, *J. Finance*, "Giving Content to Investor Sentiment") and Loughran & McDonald (2011, *J. Finance*) — media tone has incremental predictive content beyond price-derived signals at 1-4 week horizons. |
| 4 | **Insider buy ratio (30d net buying / market cap)** | `sec_provider.py` (Form 4) | Seyhun (1986, *J. Financial Economics*) and Cohen, Malloy & Pomorski (2012, *J. Finance*) — insider purchases (especially "opportunistic" buys) predict positive abnormal returns over 1-12 months. |

### Why exactly these four

- **Independence**: macro vol (VIX), macro fundamentals (yield curve), sentiment (news), private info (insiders) — minimal multicollinearity.
- **Daily-observable causally**: every one settles same-day or next-day, no look-ahead bandage needed.
- **Already in our stack**: providers exist, parallel-fetched in Phase 2 prewarm. Marginal cost of feeding them to TimesFM is one DataFrame `merge` per request.
- **Compositionally complete**: covers price/volatility/macro/private signal channels.

### Explicit non-choices

- ❌ Reddit sentiment — too noisy, low SNR at portfolio level (Antweiler & Frank 2004 finds message boards predict volatility but not direction).
- ❌ Trading volume — colinear with realized vol which TimesFM already infers from the price series.
- ❌ Google Trends — already removed from the codebase.
- ❌ EDGAR filing sentiment — strong on cross-section (Cohen, Malloy & Nguyen 2020) but very sparse temporally; better consumed as the SentimentPanel display feature, not a TimesFM covariate.

## B.4 NaN policy (no look-ahead)

For each covariate, build the aligned `[T, 4]` matrix on the same trading-day index as the target equity curve:

| Covariate | NaN handling | Rationale |
|---|---|---|
| VIX percentile rank | Forward-fill up to 3 trading days; otherwise 0.50 (median) | VIX is published daily; gaps are exchange holidays. >3d gap is a data outage — use neutral. |
| Term spread | Forward-fill up to 5 trading days; otherwise 0.0 | FRED daily but lags 1-2d on weekends; safe to ffill. |
| News sentiment EMA | NaN → 0 (neutral) | Sparse by nature; absence of news is genuinely "no signal," not missing data. |
| Insider buy ratio | NaN → 0 (no buying) | Form 4 filings are sparse; absence is informative (means no insiders bought). |

**Forbidden operations**: `shift(-1)`, `bfill`, `interpolate(method='time', limit_direction='both')`, any rolling window with `center=True`. Add a unit test that asserts the covariate matrix at row `t` doesn't depend on any data dated `> t`.

## B.5 Backend integration sketch

### New file: `backend/app/services/timesfm_provider.py`

```python
"""
TimesFM 2.5 forecast provider — self-hosted, always-warm.

Loads google/timesfm-2.5-200m-transformers once per process at startup
(via FastAPI lifespan), keeps model resident, runs CPU inference.

Zero-shot foundation-model forecasts with 4 covariates:
  VIX %ile · 10Y-2Y term spread · news sentiment EMA · insider buy ratio

References
----------
Das, A. et al. (2024). A decoder-only foundation model for time-series
  forecasting. ICML. https://arxiv.org/abs/2310.10688
"""

import logging
import threading
import time

import numpy as np
import pandas as pd
import torch
from cachetools import TTLCache
from transformers import TimesFm2_5ModelForPrediction

logger = logging.getLogger(__name__)

MODEL_ID = "google/timesfm-2.5-200m-transformers"

# Singleton model — loaded once, kept resident.
_model: TimesFm2_5ModelForPrediction | str | None = None
_load_lock = threading.Lock()

# Cache TimesFM responses keyed on (equity_curve_hash, end_date, covariate_hash).
# Same TTL as our other provider caches (1 h intraday).
_cache = TTLCache(maxsize=64, ttl=3600)


def get_model() -> TimesFm2_5ModelForPrediction | None:
    """Load TimesFM once per process. Returns None if load failed."""
    global _model
    if _model == "FAILED":
        return None
    if _model is not None:
        return _model
    with _load_lock:
        if _model is not None and _model != "FAILED":
            return _model
        try:
            t0 = time.time()
            logger.info("Loading TimesFM 2.5 (~925 MB, expect 30-60s)...")
            m = TimesFm2_5ModelForPrediction.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
            ).eval()
            _model = m
            logger.info("TimesFM loaded in %.1fs", time.time() - t0)
            return _model
        except Exception as e:
            logger.exception("TimesFM load failed — disabling for this process")
            _model = "FAILED"
            return None


def forecast_timesfm(
    returns: pd.Series,           # daily log-returns, index = trading days
    covariates: pd.DataFrame,     # [T, 4] aligned, NaN-handled per B.4
    horizon: int,                 # 63 (forward) or 30 (shadow OOS)
    last_date: str,
) -> dict:
    """
    Returns dict matching the existing MethodResult contract:
      { band: {dates, p5, p25, p50, p75, p95}, metadata: {...}, compute_ms: int }
    """
    model = get_model()
    if model is None:
        return {"error": "model_unavailable"}

    t0 = time.time()
    # TimesFM 2.5 takes a list of past_values tensors (variable lengths OK).
    # Truncate to last 1024 points if longer (model's recommended context).
    past = returns.values[-1024:].astype(np.float32)
    past_values = [torch.from_numpy(past)]

    with torch.no_grad():
        out = model(past_values=past_values, forecast_context_len=len(past))

    # out.full_predictions: (batch, horizon, num_quantiles)
    # Map TimesFM's quantile output to our p5/p25/p50/p75/p95 contract.
    # ... build band, attach metadata.compute_ms ...
```

### Wiring into `forecast_engine.py`

- Add `timesfm` to `METHOD_LABELS`, `METHOD_COLORS` (suggest `#00d4aa` — taking the freed Factor green slot).
- Add `timesfm` to `_PHASE2_SERVER_METHODS` so it joins HMM + GP in the parallel server pool.
- Submit it to `_server_pool` like HMM/GP. Pass `macro_ctx`, `news_ctx`, `insider` for covariate building (they're parallel-fetched anyway).
- Timeout budget: 30 s (inference only, no cold-load — model is always warm).

### Wiring into `main.py` (FastAPI lifespan)

```python
from contextlib import asynccontextmanager
from app.services.timesfm_provider import get_model as load_timesfm

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager-fire model load in a background thread so the first /api/forecast
    # doesn't pay the load cost. Don't await — let the app accept traffic
    # immediately; TimesFM-dependent paths gate on get_model() returning.
    asyncio.create_task(asyncio.to_thread(load_timesfm))
    yield
    # No teardown needed — process exit releases the model

app = FastAPI(lifespan=lifespan)
```

## B.6 Caching strategy

Two layers:

1. **In-process TTLCache** in `timesfm_provider.py` keyed on `(equity_hash, end_date, cov_hash)`. TTL 1 h. Critical: TimesFM inference is the most expensive step in Phase 2 (~2-10 s CPU), so cache hits matter even more than for the lightweight providers.
2. **Server-side providers** (FRED/VIX/news/insider) already cached upstream; nothing new.

Cache key for portfolio uniqueness: `hashlib.sha1(json.dumps([round(r, 6) for r in returns.tail(252)]).encode()).hexdigest()[:16]` — last 252d of returns is enough fingerprint for distinct portfolios; rounding kills float-noise misses.

## B.7 Cost notes (Railway Hobby)

Always-warm self-host has predictable monthly cost on Railway Hobby:

| Component | Cost |
|---|---|
| Hobby base subscription | $5/mo |
| Hobby credit | -$5/mo |
| Memory: ~1.8 GB always allocated × $10/GB-mo | ~$18/mo |
| vCPU: ~0.3 active vCPU avg (TimesFM bursts) × $20/vCPU-mo | ~$6/mo |
| **Net** | **~$24/mo before optimisation** |

Optimisations to cut this down:
- TTLCache hits (1h window) — likely 60-80% hit rate at low traffic, drops vCPU cost ~3-5×
- Restart service to free RAM during off-peak (manual or Railway scheduled deploy)
- If costs balloon: switch to lazy + idle-unload pattern (Phase B.2 alt) for ~$5-10/mo at the cost of cold-load latency on the first request after idle

No external quota concerns — model is local. No HF Inference rate limits to worry about.

---

# Phase C — Shared OOS holdout window

## C.1 Methodology

**The window**: hold out the trading-day window `[T−60, T−30]`, where `T` is the user's `end_date`. This gives:
- 30 days of completely held-out actual prices for shadow evaluation
- 30 days of buffer between holdout and forward forecast (avoids embargo issues per López de Prado 2018, Ch. 7)

**For each method, two separate runs per request:**

| Run | Train data | Forecast window | Used for |
|---|---|---|---|
| Shadow | `[start, T−60]` | `[T−60, T−30]` (30 trading days) | OOS R² computation, displayed against actuals |
| Forward | `[start, T]` | `[T, T+63]` | The actual forecast displayed to the user |

**Why a separate forward run** (rather than just slicing the shadow run's longer forecast): models like HMM and TimesFM behave differently when trained on `[start, T]` vs `[start, T−60]` because the most recent regime shifts inform future paths. The user wants the most-current model state for the forward forecast, even though that state is unverifiable.

## C.2 R² metric definitions

For each method, on the 30-day shadow window:

```
y_actual = portfolio_value[T−60 : T−30]               # 30 daily values
y_pred   = method.shadow_forecast.p50                  # 30 daily p50 estimates

# Convert both to log-returns (so the metric is scale-free)
r_actual = np.diff(np.log(y_actual))
r_pred   = np.diff(np.log(y_pred))

ss_res = ((r_actual - r_pred) ** 2).sum()
ss_tot = ((r_actual - r_actual.mean()) ** 2).sum()

oos_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
```

For **In-Sample R²** (already computed by some methods today, normalize across all):

```
On the 80% chronological train window of [start, T−60]:
  oos_r2 same formula, but on the held-out 20% within training
```

This means each method has a CHILD validation pass internal to the shadow training run. Most already do (HMM regime sanity, GP NLPD, LSTM val_loss); standardize all to the same R² formula on log-returns so the displayed numbers compare apples-to-apples.

## C.3 Display: shadow overlay on composite chart

Already built the composite chart's Y-axis as cumulative % return. Extension:

- Render the **shadow forecast lines** in the holdout window `[T−60, T−30]` with reduced opacity (e.g., 0.35) and a label "OOS shadow"
- Render the **actual portfolio prices** in the holdout window as a solid blue line continuation (same `CHART_COLORS.portfolio` as the historical series)
- Vertical dashed line at `T−60` ("shadow start") and `T−30` ("shadow end / actual continues") so the user sees which window is "predicted vs actual" vs "live data"
- Forward forecast lines start at `T` as today

Visually: historical → shadow window (with model overlays) → 30d actual continuation → forward forecast.

## C.4 Ensemble weighting from OOS R²

**Replace** the entire `_REGIME_WEIGHTS` machinery in [`backend/app/services/meta_learner.py`](backend/app/services/meta_learner.py) with R²-based weights.

Algorithm:

```python
def get_ensemble_weights_from_oos_r2(method_r2s: dict[str, float]) -> dict[str, float]:
    """
    Weight ensemble methods proportional to their OOS R² on the shadow window.
    TimesFM gets a 30% floor (or higher if its R² already implies more).
    """
    # Clip negative R² to 0 — a method that performs worse than constant-mean
    # contributes nothing
    clipped = {m: max(0.0, r2) for m, r2 in method_r2s.items()}
    total   = sum(clipped.values())

    if total <= 0:
        # Fallback: equal weight across methods that produced any forecast
        n = sum(1 for r2 in method_r2s.values() if r2 is not None)
        return {m: 1/n if r2 is not None else 0.0 for m, r2 in method_r2s.items()}

    weights = {m: w / total for m, w in clipped.items()}

    # TimesFM 30% floor
    TFM_FLOOR = 0.30
    if 'timesfm' in weights and weights['timesfm'] < TFM_FLOOR:
        deficit = TFM_FLOOR - weights['timesfm']
        weights['timesfm'] = TFM_FLOOR
        # Shrink the others proportionally to make room
        others = {m: w for m, w in weights.items() if m != 'timesfm'}
        other_total = sum(others.values())
        if other_total > 0:
            scale = (1.0 - TFM_FLOOR) / other_total
            for m in others:
                weights[m] = others[m] * scale

    return weights
```

**Replaces** `compute_regime_probs` + `get_ensemble_weights`. Drops `_REGIME_WEIGHTS` (4 regimes × 6 methods static dict) entirely. Drops the Ang & Timmermann (2012) citation per user instruction.

What we lose by dropping regime-conditional weights:
- The "4 regimes (bull_low_vol/bull_high_vol/bear/crisis)" decomposition. Some downstream displays (regime donut, ScenarioPanel) consume this — those need to either (a) be removed too, or (b) decouple regime classification from ensemble weighting (regime probs still come from HMM, just don't drive ensemble).
- **Recommend (b)**: HMM's regime probs are independently useful for the donut display and Kelly regime-confidence multiplier. Just stop using them for ensemble blending.

Also need to update `frontend/src/ml/MetaEnsemble.js` — drop `REGIME_WEIGHTS`, drop `blendWeights()` rule-based path, add `weightsFromR2()` mirror of the backend function (server weights are authoritative; client mirror is fallback only).

## C.5 Per-method card UI streamlining

Per user: "streamline the individual model results and focus on only 2 key metrics OOS R² and In sample R²"

Today each card shows: OOS quality label (HIGH/MED/LOW) + 1-3 method-specific metadata pills (`oos_r2`, `factor_r2`, `regime_sanity`, `ljung_box_ok`, `oos_mse`, etc. — heterogeneous).

New design: **two columns in the per-method card header — IS R² and OOS R²**, both formatted identically (e.g., `+0.082` green, `−0.014` red). Remove all the bespoke metadata pills. Move detailed metadata to an expandable "details" section.

Affected files:
- `ForecastMethodCard.jsx` — strip the `methodQuality()` and `qualityLabel()` paths; render `r.metadata.is_r2` and `r.metadata.oos_r2` directly.
- `forecast_engine.py` — every method's `metadata` must populate `is_r2` and `oos_r2`; deprecate the per-method ad-hoc fields (or keep for backward-compat in the expandable details).

---

# Phase D — Frontend changes

## D.1 Already done (commit `54b2b7b`)

- ✅ Composite Forecast Y-axis is cumulative % return (not $)
- ✅ Historical line in `CHART_COLORS.portfolio` blue, labelled "Your Portfolio"
- ✅ Zoom slider with fixed right edge

Per user "show results in terms of % gain not $" — also audit:
- `ForecastSnapshotCards.jsx` — confirm it's already showing %; if any $ values remain, convert
- `KellyPanel.jsx` — annualised return shown as %; confirmed correct after the horizon change
- `ScenarioPanel.jsx` — currently shows dollar deltas; convert to %

## D.2 Remove EnsembleDegradationStrip

The "n→5 / n→4" cap-marker logic is moot once XGBoost (the only sub-horizon method) is gone. N-BEATS' cap exactly equals the horizon.

- `ForecastComposite.jsx` — delete `EnsembleDegradationStrip` component (~40 lines), delete `capDates` array, delete the two `ReferenceLine`s for cap markers, delete `HORIZON_CAPS_DAYS` constant
- `useForecast.js` — delete `HORIZON_CAPS_DAYS` export and `capBand()` function. N-BEATS naturally produces exactly 63 days now (Phase A trims).

## D.3 Update ForecastArchitecture.jsx

Current: 8-layer pipeline diagram with `<Box>` components in rows. Update to reflect the new 5-method flow with TimesFM prominent.

Sketch (using existing `<Box>` + `<Arrow>` primitives):

```
① Inputs
[Portfolio equity curve] [Asset weights] [Date range]
  ↓
② Provider fan-out (parallel)
[FRED macro] [VIX] [News (GDELT)] [Reddit] [SEC insider] [EDGAR 10-K/Q]
  ↓
③ Two-pass evaluation (per method)
   ├─ SHADOW: train on [start, T-60], forecast [T-60, T-30] → OOS R²
   └─ FORWARD: train on [start, T],     forecast [T, T+63]  → displayed
  ↓
④ Five forecast methods (Phase 2 parallel server pool + browser)
[N-BEATS (browser)]   [HMM (server)]   [GP (server)]
[LSTM (browser)]      [TimesFM 2.5 (server, always-warm)]
  ↓
⑤ Ensemble blending (OOS R² weighted, TimesFM ≥ 30%)
[Per-method R² → normalize → TimesFM floor → blend p5/p25/p50/p75/p95]
  ↓
⑥ Outputs
[Fan chart] [Kelly sizing] [Scenario stress] [FinBERT sentiment]
```

If the box-grid gets too wide, switch to a Mermaid `flowchart TD` rendered as an `<img>` from a static SVG asset (build-time mermaid CLI), or use `react-mermaid2` if we want runtime rendering.

## D.4 Composite chart layout

Per user: "keep the composite forecast window layout but remove the ensemble active models bar at the bottom" — addressed in D.2 above.

Plus add the shadow-window overlay (Phase C.3): vertical dashed lines at `T−60` and `T−30`, model lines in `[T−60, T−30]` rendered at reduced opacity, actual portfolio prices continuing through `[T−30, T]`.

---

# Phase E — Performance + hosting checks

## E.1 Railway Hobby plan (8 GB RAM ceiling per replica, 5 GB storage)

User upgraded from free tier (512 MB) to Hobby (8 GB ceiling) specifically to support self-hosted TimesFM. After Phase A + B with always-warm TimesFM:

| Component | RAM (cold) | RAM (warm) |
|---|---|---|
| Python + FastAPI + uvicorn | ~80 MB | ~80 MB |
| pandas + numpy | ~60 MB | ~60 MB |
| hmmlearn + scipy + sklearn | ~80 MB | ~80 MB |
| statsmodels | ~40 MB | ~40 MB |
| **torch (CPU wheel)** | ~250 MB | ~280 MB |
| **transformers** | ~120 MB | ~150 MB |
| **TimesFM 2.5 weights (FP32, always-warm)** | 0 (not yet loaded) | ~925 MB |
| **TimesFM inference scratch (peak)** | 0 | ~300 MB |
| In-process caches (TTLCache × providers + TimesFM) | ~10 MB | ~40 MB |
| **Total** | **~640 MB during boot** | **~1.85 GB warm** |

Headroom under 8 GB Hobby ceiling: ~6.1 GB. Very comfortable.

Disk: TimesFM cache at `~/.cache/huggingface/hub` ≈ 925 MB. Hobby gives 5 GB → fine. If image bloat matters, **bake weights into Docker layer** (RUN command in Dockerfile downloading the model) so cold deploys don't re-fetch.

### Startup sequence

1. Container boot → uvicorn starts → app accepts traffic at ~640 MB
2. Lifespan handler dispatches `asyncio.to_thread(load_timesfm)` (background)
3. Over the next 30-60 s, weights download (first deploy only — cached after) + load into RAM, RAM grows to ~1.85 GB
4. First `/api/forecast` hit:
   - If load complete: hot path, inference 2-10 s
   - If still loading: HMM/GP/N-BEATS/LSTM all run as normal; TimesFM result reports `model_loading` and ensemble runs at 4 methods. Next request gets the full 5.

## E.2 Vercel hobby tier (100 GB bandwidth, 100 MB lambda — N/A since we're frontend-only on Vercel)

Frontend bundle audit after Phase A:

| Chunk | Before (commit 54b2b7b) | After Phase A |
|---|---|---|
| `recharts` | ~110 KB gzipped | unchanged |
| `tfjs` (LSTM) | ~280 KB gzipped | unchanged |
| `transformers` (FinBERT) | ~140 KB gzipped | unchanged |
| `markdown` | ~30 KB gzipped | unchanged |
| Main app | ~95 KB gzipped | ~88 KB gzipped (XGBoostInferer + ForecastMethodCard XGBoost block deleted) |
| `/models/xgboost/` static assets | 512 KB | 0 |

Net: ~520 KB lighter on first XGBoost forecast load. No new browser-side bundles for TimesFM (it's server-only).

## E.3 Caching keys (consolidated)

| Cache | Key | TTL | Layer |
|---|---|---|---|
| FRED macro | `(end_date, series_id)` | 24 h | `fred_provider` |
| VIX features | `(end_date)` | 24 h | `vix_provider` |
| GDELT news | `(tickers_sorted, end_date)` | 1 h | `news_provider` |
| Reddit mentions | `(tickers_sorted, end_date)` | 1 h | `reddit_provider` |
| SEC insider | `(tickers_sorted, end_date)` | 6 h | `sec_provider` |
| EDGAR filings | `(tickers_sorted, end_date)` | 24 h | `edgar_filing_provider` |
| **TimesFM forecast (NEW)** | `(equity_hash, end_date, cov_hash, horizon)` | 1 h | `timesfm_provider` |
| Forecast prewarm | fired by `prewarmForecastContext()` from frontend on backtest completion — fans out the above | n/a | `forecast.py::prewarm_forecast_context` |

## E.4 Loading speed priorities

Already in place; Phase B preserves them:
- Backend warm-up ping on App mount (`warmupBackend()`)
- Speculative provider prewarm on backtest completion (`prewarmForecastContext()`)
- Two-phase forecast: Phase 1 (~2 s) renders interactive while Phase 2 (~15-25 s) computes
- Lazy-loaded ForecastPanel via `React.lazy`

New consideration for Phase C (shadow runs double the compute):
- The shadow runs are independent of the forward runs and can fire in parallel
- For N-BEATS / LSTM (browser): two separate web-worker-like dispatches
- For HMM / GP / TimesFM (server): four-way parallelism via the existing `_server_pool` (bump max_workers from 2 → 4)
- Net wall-clock: Phase 2 stays at ~15-25 s warm because the parallelism absorbs the doubled compute

## E.5 Dead-code sweep checklist (post all phases)

```bash
# Backend
grep -rn "factor\|xgboost\|FF5\|REGIME_WEIGHTS" backend/app/ \
  | grep -v "_pycache\|test_\|sec_provider\|insider\|factor_decomposition.py"
# Should return: only acceptable matches (insider Form 4 wording, etc.)

# Frontend
grep -rn "xgboost\|XGBoost\|factor\|Factor.*Model\|REGIME_WEIGHTS\|HORIZON_CAPS_DAYS" frontend/src/
# Should return: zero forecast-engine references

# Cleanup verification
ls frontend/public/models/  # → lstm, nbeats (no xgboost)
ls backend/scripts/         # → no train_xgboost.py
```

Watch for orphaned imports, broken React component prop chains, and any remaining `var/VAR` references that should now be unambiguously `gp/GP` (legacy "var" key was kept for backward-compat — consider whether to migrate that key while we're already breaking the API contract).

---

# Implementation order (recommended)

Don't try to land this all at once. Suggested PR sequence:

1. **PR-1: Drop XGBoost + Factor (Phase A)** — pure deletion, low risk, isolated. Verify backend + frontend still build and forecast endpoint succeeds with the 4 remaining methods.
2. **PR-2: Add TimesFM provider, no UI (Phase B.1-B.6)** — wire `timesfm_provider.py` (always-warm self-host), add `transformers` + `torch` to `requirements.txt`, FastAPI lifespan eager-load, add to method dispatch, but don't yet add a method card or change the chart. Verify `/api/forecast` returns a `timesfm` MethodResult. Test locally first (model downloads to `~/.cache/huggingface/hub`), then deploy to Railway Hobby and confirm RAM stays under 2.5 GB peak.
3. **PR-3: Shadow holdout evaluation (Phase C.1-C.3)** — modify all method functions to accept a `train_until` parameter, run shadow + forward, compute OOS R². Display on the chart. Backend-heavy.
4. **PR-4: R²-weighted ensemble (Phase C.4)** — replace `meta_learner.py` regime path with R² weighter. Frontend `MetaEnsemble.js` mirror. UI displays new weights.
5. **PR-5: Per-method card streamlining (Phase C.5)** — IS R² + OOS R² columns, kill bespoke pills. Cosmetic.
6. **PR-6: Architecture diagram + EnsembleDegradationStrip removal (Phase D)** — final polish.
7. **PR-7: Dead-code sweep + memory + cache audit (Phase E)** — verification commit, no behaviour change.

Each PR should:
- Update `context-log.md` with what changed and why
- Bump the visible commit version badge (already wired)
- Have a manual smoke test note in the commit message

---

# Open questions / decisions to resolve at implementation time

1. ~~**TimesFM 2.5 model ID and exact API contract**~~ — RESOLVED 2026-05-01: `google/timesfm-2.5-200m-transformers`, class `TimesFm2_5ModelForPrediction`, Apache 2.0, not gated.
2. ~~**HF Inference API auth scope**~~ — RESOLVED: HF Serverless does not support `time-series-forecasting` pipeline. We self-host instead. No HF token required for inference.
3. **What to do with `var` method key** — backward-compat alias for GP. Now that we're breaking the contract anyway (dropping xgboost+factor, adding timesfm), should we rename `var` → `gp` end-to-end? Recommend yes; cheaper than carrying the alias forever.
4. **Backward-compat for old saved AI conversations** — if any users have an in-flight AI chat that referenced a constructed portfolio with old method names, the next render call may fail. Low risk (rate-limited, no persistent sessions), but worth noting.
5. **Regime-conditional UI displays** — `ScenarioPanel.jsx` and `EnsembleCard.jsx` regime donut still consume `regime_probs`. Decision: keep these (regime classification is independently useful), just sever the regime → ensemble weighting linkage.

---

# Out of scope (do NOT do as part of this upgrade)

- Renaming `var` → `gp` end-to-end (decide separately per Q3 above)
- Removing `arch` / `statsmodels` deps (premature; defer until we have proof they're unused)
- Adding more covariates beyond the 4 listed in B.3 (additive complexity without evidence; revisit if TimesFM OOS R² is poor)
- Replacing N-BEATS with N-HiTS or another newer model (TimesFM already covers the foundation-model role)
- Regime-conditional ensemble re-introduction (specifically asked to remove)

---

# References

### Foundation model
- Das, A., Kong, W., Sen, R., & Zhou, Y. (2024). *A decoder-only foundation model for time-series forecasting*. ICML. https://arxiv.org/abs/2310.10688

### Covariates
- Bekaert, G. & Hoerova, M. (2014). The VIX, the variance premium and stock market volatility. *Journal of Econometrics*, 183(2), 181–192.
- Estrella, A. & Hardouvelis, G. (1991). The term structure as a predictor of real economic activity. *Journal of Finance*, 46(2), 555–576.
- Ang, A., Piazzesi, M. & Wei, M. (2006). What does the yield curve tell us about GDP growth? *Journal of Econometrics*, 131(1-2), 359–403.
- Tetlock, P. (2007). Giving content to investor sentiment. *Journal of Finance*, 62(3), 1139–1168.
- Loughran, T. & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65.
- Seyhun, H.N. (1986). Insiders' profits, costs of trading, and market efficiency. *Journal of Financial Economics*, 16(2), 189–212.
- Cohen, L., Malloy, C. & Pomorski, L. (2012). Decoding inside information. *Journal of Finance*, 67(3), 1009–1043.

### Methodology
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 7 (Cross-Validation in Finance — purged k-fold, embargo).

### Removed
- ~~Ang, A. & Timmermann, A. (2012). Regime Changes and Financial Markets. *Annual Review of Financial Economics*, 4, 313–337.~~ — was the basis for `_REGIME_WEIGHTS`; dropped per user instruction.
