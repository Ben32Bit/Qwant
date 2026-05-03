# TimesFM 2.5 Financial Fine-Tuning — Plan & Reference

Status: **planned, not implemented** as of 2026-05-02.
Owner: project owner.
Audience: future Claude sessions, future contributors.

This document is the canonical reference for fine-tuning the TimesFM 2.5
forecaster used by `backend/app/services/timesfm_provider.py`. Read it
before touching that file or starting any training run.

---

## 1. Problem

The current zero-shot `google/timesfm-2.5-200m-pytorch` produces conservative
mean-of-history forecasts on equity portfolios. On flat shadow holdout
windows it scores OOS R² in the −80 to −100 range, far worse than the other
4 models in our ensemble (HMM, GP, N-BEATS, LSTM). The model is technically
sound — it's just trained on diverse non-financial series (sales, energy,
weather) and treats financial data with the same regression-to-mean prior.

**Goal**: a fine-tuned variant that beats zero-shot 2.5 on shadow OOS R²
across a held-out S&P 500 sample, deployable on the same Railway hobby-tier
container without OOM or latency regression.

## 2. Reference paper

> Fu, X., Hirano, M., & Imajo, K. (2024). *Financial Fine-tuning a Large
> Time Series Model.* arXiv:2412.09880. https://arxiv.org/abs/2412.09880
> Code: https://github.com/pfnet-research/timesfm_fin

**Headline finding**: continual pre-training of TimesFM 1.0 on ~90M
financial price points lifted market-neutral S&P 500 Sharpe (h=128) from
0.42 (zero-shot) to 1.68. Zero-shot was below random chance on 4/7 horizons.

**Key tricks**:
- Input is `log(price)`, not returns — handles crashes without NaNs and
  stays scale-invariant via TimesFM's per-window normalisation.
- Loss is **MSE on log-prices**.
- Optimizer is plain SGD+momentum; full continual pre-training (no LoRA).
- Authors flag in §VI: full FT may have destroyed general-TS capability;
  LoRA on attention would have been preferable; quantile loss would have
  been preferable.

**Why we can't use their checkpoint as-is**: it's TimesFM **1.0** (JAX, no
quantile head, smaller context). `transformers.TimesFm2_5ModelForPrediction`
expects a different state dict. Forcing 1.0 weights would mean abandoning
our stable HF integration.

## 3. Architectural constraint: univariate only

`TimesFm2_5ModelForPrediction` is **univariate** — `past_values` is
`(batch, seq_len)`, one scalar per timestep. There is no native covariate
channel.

**Implication for fine-tuning**: we cannot fuse macro / news / sentiment /
volume directly into the model. We expand the *universe* of univariate
series instead. Non-price features stay in our existing `forecast_engine.py`
ensemble layer (which already consumes macro, FRED, news, Reddit, EDGAR via
the Phase 2 server-side methods).

## 4. Why log-prices for training (and not in zero-shot inference)

This is the most likely confusion point — we explicitly dropped log-prices
from the zero-shot inference path two commits ago. They're still right
for training. The reasons are different.

### 4.1 What we tried in zero-shot inference

| Approach | Commit | Result |
|---|---|---|
| Normalized levels (end = 1.0) | original | Mean-of-history bias, OOS R² ≈ −100 |
| **Log-prices + `exp()` conversion** | `ef16507` | **Blown up** — `exp()` amplified zero-shot output drift into multi-thousand-% spikes |
| Raw equity levels + linear conversion | `ac091fe` (current) | Stable; bounded by linear math |

The blowup happened because the **zero-shot** model wasn't trained to output
log-prices in any specific range. Small drifts in its outputs got
exponentially amplified by the `exp()` conversion.

### 4.2 Why log-prices are right for training

The paper trains the model **to output log-prices in the correct range**.
Training data is log-prices of stocks (typically `[1, 8]` in log-space for
$2-$3000 prices). MSE loss penalizes outputs outside that range. After
training, model outputs are calibrated — no `exp()` blowup risk.

Log-price training also fixes two pathologies the paper documents (their
§III-B):
- Raw-price MSE biases hard toward expensive stocks ($1000 stock dominates
  loss vs $5 stock by a factor of 200²)
- Raw-price training NaNs out on bankruptcies / extreme single-day moves;
  `log(0)` is `-inf` but log of any positive price is finite

### 4.3 Inference path branches

After fine-tuning, `forecast_timesfm()` will have two code paths:

| When | Input transform | Output conversion |
|---|---|---|
| `_adapter_id is None` (zero-shot fallback) | Raw equity levels | Linear: `(pred - L_T) / L_T` |
| `_adapter_id is set` (fine-tuned, default) | `log(equity_curve)` | `exp(pred - log(L_T)) - 1` |

The fine-tuned exp path also gets a **safety clamp**: clip predicted
log-prices to within `±0.7` log-units of the input's last log-price (≈ ±100%
cumulative return cap over the 30-day horizon). Belt-and-suspenders against
any tail behaviour the LoRA adapter doesn't fully tame.

## 5. Method

### 5.1 LoRA on attention, not full FT

| | Choice |
|---|---|
| Adapter type | **LoRA** via `peft` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention only) |
| Rank | `r = 16`, `alpha = 32`, `dropout = 0.05`, `bias = "none"` |
| Frozen | All MLPs + base attention weights |
| Adapter size on disk | ~25 MB |

Justification:
- Paper §VI explicitly recommends it.
- 25 MB adapter vs 800 MB full checkpoint — fits git, fits 5 GB Railway
  volume comfortably, ships in seconds via HF Hub.
- Avoids catastrophic forgetting of general-TS capability (paper's flagged
  failure mode).
- `peft.PeftModel` lets us toggle adapter on/off in-process; 30-50 MB extra
  RAM only.

Verify exact attention module names with `model.named_modules()` in a
notebook before launching — TimesFM 2.5's HF naming may differ from
canonical Llama-style names.

### 5.2 Loss

Primary: **MSE on log-prices** (paper recipe, proven).

Secondary attempt (if quick to wire up): **quantile pinball loss** at
`[0.1, 0.2, …, 0.9]` against log-price targets. TimesFM 2.5's HF forward
returns `quantile_preds` natively. Paper's §VI explicitly recommends this
as a follow-up.

If quantile loss is awkward to wire (HF build's quantile path is fiddly —
see prior notes in `timesfm_provider.py`), ship MSE-on-log-price for v1.
Document which one shipped in `metadata.train_loss`.

### 5.3 Hyperparameters

| Param | Value | Source |
|---|---|---|
| Optimizer | AdamW, lr 1e-4, weight_decay 0.01 | adjusted from paper (SGD+momentum) — AdamW more reliable with LoRA |
| Schedule | linear warmup 500 steps → cosine decay over 30 epochs | adjusted from paper (25 warmup / 100 cosine) — LoRA converges faster |
| Batch | 256 (grad accum to 1024 if VRAM allows) | adjusted from paper (1024) |
| Context | random uniform `[256, 1024]` per batch | extended from paper (128-512) — 2.5 supports much longer |
| Horizon | 128 | paper |
| Precision | bf16 on L40S/A100 | |
| Grad clip | max-norm 1.0 | paper |

## 6. Data: full universe from the start

Single-shot training on the diverse universe — no MVP / phased data ramp.
Larger and more varied training data → better regime coverage and broader
asset-class robustness. yfinance handles all of it.

### 6.1 Universe (~1,000-1,100 tickers)

| Group | Approx count | Examples |
|---|---|---|
| S&P 500 constituents | ~500 | AAPL, MSFT, JPM, … |
| Russell 1000 ex-S&P 500 (small/mid caps) | ~500 | mid-cap US equities not in S&P 500 |
| Sector ETFs | 11 | XLK, XLF, XLE, XLV, XLI, XLP, XLY, XLU, XLB, XLRE, XLC |
| Broad-market ETFs | ~10 | SPY, QQQ, IWM, VTI, EFA, EEM, VTV, VUG, MDY, SCHD |
| Bond ETFs | ~6 | TLT, IEF, AGG, HYG, LQD, SHY |
| Commodity ETFs | ~5 | GLD, USO, SLV, DBA, DBC |
| FX ETFs | ~3 | UUP, FXE, FXY |
| Volatility | 1 | VIX as a level series (`^VIX` in yfinance) |

All stay **univariate** — each series fine-tunes the same 200M model
independently. No multivariate fusion.

### 6.2 Skipped data sources

| Skip | Reason |
|---|---|
| News sentiment | Sparse, noisy, no native fusion path in TimesFM 2.5 |
| Reddit sentiment | Recent-only history — leakage risk into 2024+ user backtests |
| EDGAR / SEC filings | Per-ticker async events; no fusion path |
| Insider trades | Same fusion problem |
| Volume | Adds little vs price alone for univariate model; not worth complexity |
| Crypto | Paper saw weak FT results on crypto; out of our user-base scope |
| FX cross-pairs | Out of user-base scope; paper saw mixed results |

These remain consumed by the existing `forecast_engine.py` ensemble layer
(Phase 2 server methods + N-BEATS macro context).

### 6.3 Train / val / test split

| Split | Period | Tickers | Use |
|---|---|---|---|
| Train | 2000-01-01 → 2022-12-31 | 90% of universe (random by ticker) | gradient updates |
| Val | 2000-01-01 → 2022-12-31 | 10% holdout tickers | early stop |
| **Test** | **2023-01-01 → 2023-12-31** | 50 random S&P 500 tickers | **deploy-blocking eval** |

**Train cutoff is 2022-12-31, hard.** 2024-2026 is reserved for users'
backtests — any leakage there would silently inflate every shadow OOS R² in
production.

### 6.4 Preprocessing

1. `log(adj_close)` per ticker
2. Drop tickers with <1000 valid days
3. Per-ticker, slide windows: random start, length sampled uniform from
   `[256, 1024]`, target = next 128 days. Skip windows containing any NaN.
4. Sample ~150 windows per ticker per epoch → ~150k windows/epoch,
   ~600 steps/epoch at batch 256

### 6.5 Storage format

Single Parquet file `~/timesfm_ft_data/universe_logprices_2000_2022.parquet`,
columns `[ticker, date, log_close]`. ~80-100 MB on disk for the full
~1,100-ticker universe. Built once on the local machine from yfinance,
cached locally — don't re-pull every run.

Pre-download locally and `scp` to the rented GPU rig — yfinance over
RunPod's network is slow and rate-limited.

**Universe construction script** (`backend/scripts/build_finetune_universe.py`,
new) snapshots:
- S&P 500: scrape Wikipedia or use static `sp500.csv`
- Russell 1000: static `russell1000.csv` (one-shot snapshot)
- ETFs: hard-coded list
- VIX: `^VIX`

Snapshot files committed to repo so the run is reproducible.

## 7. Compute plan: RunPod

| Option | Cost | Wall-clock | Verdict |
|---|---|---|---|
| Railway direct | $0 | 5-14 days CPU | impossible — no GPU |
| Colab free (T4 16 GB) | $0 | 5-8 h | fallback if RunPod auth blocked |
| Colab Pro (T4/L4) | $10/mo | 3-5 h | fallback |
| **RunPod L40S 48 GB** | **$0.79-0.99/hr** | **2-3 h** | **primary** |
| RunPod A100 40 GB | $1.50-2.50/hr | 1.5-2 h | overkill for LoRA |

Bigger universe (~1,100 tickers vs the original phased ~600) increases
wall-clock by ~50% — still well under one rental session.

**Total compute cost: ~$5-8 one-off.**

### Launch recipe (RunPod)

```bash
# On RunPod L40S 48 GB, "PyTorch 2.4 + CUDA 12.4" template, 100 GB disk
git clone <our repo>
cd Qwant
pip install -r backend/requirements-train.txt
export HF_TOKEN=...                     # if TimesFM 2.5 is gated
# Pre-built parquet uploaded via scp to /workspace/Qwant/data/
python -m backend.scripts.finetune_timesfm \
    --output ./adapter_out \
    --epochs 30 \
    --rank 16 \
    --batch-size 256

# Locally
scp -P <port> root@<pod-ip>:/workspace/Qwant/adapter_out/* ./local/
```

## 8. Server-side integration

### 8.1 Storage

**HF Hub private repo**: `qwant/timesfm-2.5-fin-lora-v1`.
- Versioned (`v1`, `v2` tags)
- Free CDN, `from_pretrained()` API
- Cached in `~/.cache/huggingface/` on Railway — same volume as base model

Fallback: commit to `backend/models/timesfm_lora_v1/` in git (25 MB is OK
as a one-shot, but bumps next iteration go to Hub).

### 8.2 Loader changes (`timesfm_provider.py`)

```python
def load_timesfm() -> None:
    ...
    model = cls.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    if os.environ.get("TIMESFM_FT_ENABLED") == "true":
        adapter_id = os.environ.get(
            "TIMESFM_FT_ADAPTER", "qwant/timesfm-2.5-fin-lora-v1"
        )
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model, adapter_id, token=os.environ.get("HF_TOKEN")
            )
            global _adapter_id
            _adapter_id = adapter_id
        except Exception as exc:
            logger.warning("FT adapter failed, falling back to base: %s", exc)
    model.eval()
    _model = model
```

### 8.3 Inference branching in `forecast_timesfm()`

```python
def forecast_timesfm(returns, horizon, last_date):
    ...
    levels = (1.0 + returns).cumprod()

    if _adapter_id is not None:
        # Fine-tuned path: log-prices in, exp(pred-last_log)-1 out
        context = np.log(levels.values)[-CONTEXT_LEN:].astype(np.float32)
        last_anchor = float(np.log(levels.iloc[-1]))
        # ... model call ...
        # Safety clamp: ±0.7 log-units ≈ ±100% cum return cap
        clamped = np.clip(pred_logs - last_anchor, -0.7, 0.7)
        cum_ret_pct = (np.exp(clamped) - 1.0) * 100.0
    else:
        # Zero-shot fallback: raw levels in, linear out (current production)
        context = levels.values[-CONTEXT_LEN:].astype(np.float32)
        last_anchor = float(levels.iloc[-1])
        # ... model call ...
        cum_ret_pct = ((pred_levels - last_anchor) / last_anchor) * 100.0
```

Cache key in `_cache: TTLCache` must include `_adapter_id` so swapping
adapter versions doesn't serve stale forecasts.

### 8.4 Memory budget on Railway hobby

| Component | RAM |
|---|---|
| Base TimesFM 2.5 fp32 | ~1.7-2.1 GB |
| LoRA adapter via `peft` | +30-50 MB |
| FastAPI + numpy/pandas/uvicorn overhead | ~700-900 MB |
| **Total working set** | **~2.5-3.0 GB** |

Default 8 GB Hobby cap is comfortable. Recommend bumping to 12 GB in
service settings as safety margin (free config knob).

### 8.5 API contract

**Unchanged.** `forecast_timesfm(returns, horizon, last_date)` keeps the
same signature and `{band, metadata, compute_ms}` return.

New `metadata` fields:
- `adapter_id`: `"qwant/...lora-v1"` or `None`
- `train_cutoff`: `"2022-12-31"` (audit trail for leakage)
- `train_loss`: `"mse_log_price"` or `"quantile_pinball"`
- `input_transform`: `"log_price"` or `"raw_level"` (matches branch above)
- `finetuned`: `True` / `False`

### 8.6 Failure modes

- Adapter download fails → log warning, run base model on raw-level path,
  no 5xx
- Adapter loads but inference NaNs → caught by existing TTLCache+exception
  path in `forecast_engine.py`, method shows error in UI
- A/B revert: flip `TIMESFM_FT_ENABLED=false` env var on Railway, no
  redeploy needed

## 9. Validation strategy

### 9.1 Pre-deployment eval (deploy-blocking)

Last step of `finetune_timesfm.py`:

For each of 50 random S&P 500 tickers, take 2023 daily returns, run the
existing `forecast_timesfm` shadow path (train-on-first-N, forecast-30,
compare to actual), once with base model (raw-level path), once with
adapter (log-price path). Reuse `forecast_engine.py`'s shadow logic to
keep evaluation apples-to-apples with production.

Output: `backend/scripts/finetune_eval_v1.json` with per-ticker R² delta,
mean, median, distribution histogram, and sample diagnostic plots.

### 9.2 Go / no-go thresholds

- **Median OOS R² delta ≥ +0.02** (absolute lift; zero-shot is often near 0,
  so +2pp median is meaningful)
- **≥ 60% of tickers improve**
- **No catastrophic regression**: ≥ 95% of tickers within −0.05 of base
- **Latency parity**: adapter inference within +20% of base on Railway

If gate fails: do not commit env var change. Iterate on hyperparams
(try r=8, fewer epochs, swap loss).

### 9.3 Production monitoring

Existing shadow OOS R² is shown per-method in the UI. Regressions appear
in production immediately — no extra instrumentation needed.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Catastrophic forgetting (paper §VI) | LoRA r=16, attention-only, base frozen |
| `exp()` blowup at inference | Output clamp at ±0.7 log-units; LoRA training calibrates outputs into the correct range |
| Overfitting to 2015-2022 regime (low rates) | 2023 holdout in deploy gate (rate-hike regime) |
| Train data leaking into user backtests | Hard 2022-12-31 cutoff, surfaced via `metadata.train_cutoff` |
| Memory OOM on Railway | Bump soft cap to 12 GB; LoRA adds <100 MB so this is precaution |
| Latency regression | Pre-deploy benchmark + shadow R² monitoring |
| TimesFM license | Apache 2.0 on model card — re-verify before publishing adapter |
| HF rate limit / repo gate | Pre-cache `HF_TOKEN` on Railway; pin `HF_HUB_CACHE` to volume |
| Adapter download fails on boot | Already handled — graceful fallback to base raw-level path |
| Russell 1000 data quality (yfinance gaps for small caps) | Drop tickers with <1000 valid days during preprocessing |

## 11. Single-shot rollout

No phasing. One training run on the full ~1,100-ticker universe, one
adapter, one deploy gate.

If the deploy gate fails:
- Iterate on hyperparams locally on a smaller universe (S&P 100 subset)
- Re-run full training only when small-universe iteration shows a lift

If subsequent improvements are wanted:
- New training runs versioned as `v2`, `v3` adapters on HF Hub
- Bump `TIMESFM_FT_ADAPTER` env var; old adapter kept available for rollback

## 12. File map

| File | Status | Purpose |
|---|---|---|
| `backend/scripts/finetune_timesfm.py` | **NEW** | Main training script |
| `backend/scripts/build_finetune_universe.py` | **NEW** | Build the ~1,100-ticker parquet from yfinance |
| `backend/scripts/sp500_constituents.csv` | **NEW** | Static S&P 500 snapshot |
| `backend/scripts/russell1000_constituents.csv` | **NEW** | Static Russell 1000 snapshot |
| `backend/requirements-train.txt` | **NEW** | `peft>=0.13`, `accelerate>=1.0`, `datasets>=3.0` |
| `backend/scripts/finetune_eval_v1.json` | **NEW** (output of training) | Deploy gate eval results |
| `backend/app/services/timesfm_provider.py` | modified | PEFT load, adapter metadata, branched inference path, output clamp |
| `backend/app/services/forecast_engine.py` | unchanged | Shadow OOS R² already auto-measures lift |
| `backend/scripts/train_lstm.py` | reference only | Pattern to mirror — config block, validation gate, `if __name__ == "__main__"` |

## 13. Open decisions

1. **HF Hub vs git** for the adapter — Hub is cleaner; git is zero-ops if
   you don't want another account. Adapter is 25 MB either way.
2. **MSE-on-log-price vs quantile pinball loss** — start with MSE
   (paper-faithful, simpler). If trained model converges and HF quantile
   path works in `forecast_timesfm`, swap to pinball for a v2 adapter.
3. **RunPod sign-up timing** — requires a credit card. If deferring,
   Colab Pro ($10/mo) works with smaller batch.
4. **Output clamp range** — proposed ±0.7 log-units (≈ ±100% cum return).
   Tighter (±0.5 ≈ ±65%) is safer but may clip legitimate large moves
   in vol regimes.

## 14. Estimated total work

- Engineer hours: **12-16 h** for the single-shot rollout
- One-off compute: **~$5-8 RunPod**
- Recurring monthly cost on Railway: **$0**

## 15. References

- Fu, Hirano, Imajo (2024). Financial Fine-tuning a Large Time Series
  Model. arXiv:2412.09880. https://arxiv.org/abs/2412.09880
- pfnet-research/timesfm_fin — code & TimesFM 1.0 checkpoint.
  https://github.com/pfnet-research/timesfm_fin
- Das et al. (2024). A decoder-only foundation model for time-series
  forecasting (original TimesFM). ICML 2024. arXiv:2310.10688
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
  arXiv:2106.09685
- HuggingFace TimesFM 2.5 model card:
  https://huggingface.co/google/timesfm-2.5-200m-pytorch
- Railway pricing & limits: https://railway.com/pricing
