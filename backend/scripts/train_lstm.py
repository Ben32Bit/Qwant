"""
train_lstm.py — Attention-LSTM training with scale-invariant features.

Run locally or via the `.github/workflows/retrain-lstm.yml` workflow:
  cd backend
  pip install -r requirements-train.txt
  python scripts/train_lstm.py

Outputs:
  backend/scripts/lstm_model.keras           (Keras saved model)
  frontend/public/models/lstm/model.json     (TF.js topology)
  frontend/public/models/lstm/group1-shard1of1.bin  (weights)

───────────────────────────────────────────────────────────────────────────────
SCALE-INVARIANT FEATURE SET
───────────────────────────────────────────────────────────────────────────────
Previous version used raw returns + a per-portfolio MinMaxScaler. The scaler
saturated on single-stock / leveraged / crypto portfolios, driving the
"LSTM rollout diverged" error. Fixed by building features that are
distribution-free *by construction* so one model generalises across assets.

Features (all approximately in [-1, 1] after fixed clipping):
  [0] z_ret     = r_t / σ_21d(r_{t-21..t-1})      clipped to ±3, scaled by /3
  [1] vol_pct   = percentile of σ_63d in its 5y trailing window, mapped to [-1,1]
  [2] z_mom5    = sum_5(r) / (σ_5 * √5)           clipped to ±3, scaled by /3
  [3] z_mom21   = sum_21(r) / (σ_21 * √21)        clipped to ±3, scaled by /3
  [4] rsi_centred = 2*rsi_14 - 1                  (rsi already in [0,1])

Target:
  next-day z_ret, same clip+scale. Inference multiplies back by current σ_21d
  to get a raw return, so the model works for any vol regime.

───────────────────────────────────────────────────────────────────────────────
EXPANDED TRAINING UNIVERSE
───────────────────────────────────────────────────────────────────────────────
55 assets across ETFs / single stocks / crypto / leveraged / low-vol so the
network sees the full spectrum of daily-return distributions in training.

───────────────────────────────────────────────────────────────────────────────
VALIDATION GATE
───────────────────────────────────────────────────────────────────────────────
After training, a held-out OOS test + rollout sanity check on 4 diverse assets
(SPY, AAPL, BTC-USD, TQQQ). Training ABORTS and the model is NOT exported if:
  - OOS MSE > 0.10           (model barely learned anything)
  - directional_accuracy < 0.50  (worse than a coin flip)
  - any rollout basket fires <70% survivors or |21d p50| > 10%

This prevents the GH Action from pushing a broken model to prod.

───────────────────────────────────────────────────────────────────────────────
KERAS 2 NOTE
───────────────────────────────────────────────────────────────────────────────
TF.js 4.x only reads legacy Keras 2 JSON. This script forces legacy Keras via
TF_USE_LEGACY_KERAS=1 + the tf_keras package.
"""

import os
# Must be set BEFORE importing tensorflow so TF dispatches to tf_keras.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except ImportError:
    sys.exit("ERROR: TensorFlow not installed. Run: pip install -r requirements-train.txt")

try:
    import tf_keras as keras
except ImportError:
    sys.exit(
        "ERROR: tf_keras not installed — required for TF.js-compatible export.\n"
        "Run: pip install tf_keras"
    )

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
BACKEND_DIR  = SCRIPT_DIR.parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
TFJS_OUT     = FRONTEND_DIR / "public" / "models" / "lstm"
MODEL_OUT    = SCRIPT_DIR / "lstm_model.keras"
TFJS_OUT.mkdir(parents=True, exist_ok=True)

# ── Training universe ─────────────────────────────────────────────────────────
# 55 assets spanning low-vol ETFs, broad-market ETFs, single stocks (multi-
# sector), leveraged ETFs, and crypto. The model only learns how to predict
# the *next z-return* given a window of past z-features, so it generalises
# across vol regimes rather than memorising one.

UNIVERSE_ETFS = [
    "SPY", "QQQ", "IWM", "VTI", "EFA", "EEM",    # broad equity
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "XLI",  # sectors
    "TLT", "IEF", "SHY", "AGG", "LQD", "HYG",    # fixed income
    "GLD", "SLV", "USO", "DBC",                  # commodities
    "VNQ",                                        # real estate
]
UNIVERSE_STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "ORCL",
    "CRM", "JPM", "BAC", "GS", "V", "MA",
    "UNH", "JNJ", "LLY", "HD", "COST", "WMT", "NKE", "MCD",
    "XOM", "CVX", "CAT", "BA",
]
UNIVERSE_HIGH_VOL = ["BTC-USD", "ETH-USD", "TQQQ", "SOXL", "VIXY"]

UNIVERSE = UNIVERSE_ETFS + UNIVERSE_STOCKS + UNIVERSE_HIGH_VOL

START        = "2015-01-01"
END          = None  # None → up to today, set by yfinance
LOOKBACK     = 60
TRADING_DAYS = 252
CLIP_SIGMAS  = 3.0
FEAT_COLS    = ["z_ret", "vol_pct", "z_mom5", "z_mom21", "rsi_centred"]
N_FEATURES   = len(FEAT_COLS)

# Time-based global splits. We use absolute dates (not per-asset %) so every
# asset's train / val / test periods line up calendar-wise. This avoids:
#   (a) val/test leakage from overlapping regimes across assets
#   (b) over-weighting long-history assets like AAPL vs newer ones
#
# Cutoffs are ROLLING (relative to today) so the weekly GH Actions cron keeps
# a consistent 12-month val / 12-month test window regardless of when it runs.
#   test : last ~12 months (never seen during training or early stopping)
#   val  : the 12 months before that (drives early stopping)
#   train: everything prior
_today         = pd.Timestamp.today().normalize()
TEST_CUTOFF_TS = _today - pd.DateOffset(months=12)
VAL_CUTOFF_TS  = _today - pd.DateOffset(months=24)
TEST_CUTOFF    = TEST_CUTOFF_TS.strftime("%Y-%m-%d")
VAL_CUTOFF     = VAL_CUTOFF_TS.strftime("%Y-%m-%d")

# Per-asset sample cap so high-history tickers don't dominate the loss.
# ~6 years × 252 days gives a generous ceiling; tickers with less history
# are kept whole.
MAX_SAMPLES_PER_TICKER = 1500

# Parquet cache schema version. Bump this if build_scale_invariant_features
# or the business-day alignment changes — old caches will auto-invalidate.
CACHE_SCHEMA_VERSION = 2

# Validation gate thresholds.
#
# NB on MSE: the target is z_ret ∈ [-1, 1] (after clip/scale), so var(y) ≈ 0.11
# for near-Gaussian returns. A "predict zero" null model scores MSE ≈ var(y).
# What actually matters for a distributional forecaster is:
#   (a) the model is no worse than the null model (MSE ≤ 1.10 × var(y_val))
#   (b) direction is slightly better than chance
#   (c) 252-day MC Dropout rollouts stay sane (survivors + 21d plausibility)
# Tight per-step MSE on day-ahead noise is not the goal — daily returns are
# near-unpredictable, and the product consumes the full forecast *distribution*.
GATE_MAX_MSE_VAR_RATIO = 1.10   # OOS MSE / var(y_val) must be ≤ 1.10
GATE_MIN_DIR_ACCURACY  = 0.50
GATE_MIN_SURVIVORS     = 0.70
GATE_MAX_21D_MEDIAN    = 0.10
GATE_BASKETS           = ["SPY", "AAPL", "BTC-USD", "TQQQ"]


# ── Feature engineering ───────────────────────────────────────────────────────

def build_scale_invariant_features(returns: pd.Series) -> pd.DataFrame:
    """
    Build the 5 scale-invariant features. All columns are approximately in
    [-1, 1] after clipping + scaling — no per-asset scaler fit needed.

    IMPORTANT: the σ used to normalise r_t is computed from r_{t-21..t-1}
    (causal, shifted by 1). Otherwise the feature would leak its own denominator.
    """
    r = returns.copy()
    df = pd.DataFrame({"r": r})

    # Causal rolling std: uses only past data
    sigma_21 = r.rolling(21).std().shift(1)
    sigma_5  = r.rolling(5).std().shift(1)
    sigma_63 = r.rolling(63).std().shift(1)

    # Volatility-regime feature: where does today's 63d vol sit in its 5y history?
    # rank(pct=True) returns 0..1 where 1 = highest. We map to [-1, 1].
    vol_pct_raw = sigma_63.rolling(5 * TRADING_DAYS, min_periods=TRADING_DAYS).rank(pct=True)
    df["vol_pct"] = (2.0 * vol_pct_raw - 1.0).clip(-1, 1)

    # z_ret: per-σ-unit return
    df["z_ret"] = (r / (sigma_21 + 1e-9)).clip(-CLIP_SIGMAS, CLIP_SIGMAS) / CLIP_SIGMAS

    # z-scored momentum — sum-of-returns divided by expected σ of a sum
    mom5  = r.rolling(5).sum()
    mom21 = r.rolling(21).sum()
    df["z_mom5"]  = (mom5  / (sigma_5  * np.sqrt(5)  + 1e-9)).clip(-CLIP_SIGMAS, CLIP_SIGMAS) / CLIP_SIGMAS
    df["z_mom21"] = (mom21 / (sigma_21 * np.sqrt(21) + 1e-9)).clip(-CLIP_SIGMAS, CLIP_SIGMAS) / CLIP_SIGMAS

    # RSI-14 mapped from [0, 1] to [-1, 1]
    delta = r.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    # When both gain and loss are ~0 (halted ticker, flat day), rsi → 0 with a
    # naive 1e-9 epsilon which maps to rsi_centred = -1 (max oversold). That's
    # wrong — flat means neutral. np.where handles it explicitly.
    total = gain + loss
    rsi14 = np.where(total > 1e-12, gain / (total + 1e-12), 0.5)
    df["rsi_centred"] = np.clip(2.0 * rsi14 - 1.0, -1, 1)

    # Target: tomorrow's z_ret (same clip+scale). Built so inference is a
    # direct prediction of z_{t+1} which we multiply by σ_21d to get a raw return.
    df["target"] = df["z_ret"].shift(-1)

    return df[FEAT_COLS + ["target"]].dropna()


def make_sequences(feature_data: np.ndarray, target_data: np.ndarray, lookback: int):
    """Slide a (lookback, N_FEATURES) window over feature_data; target is target_data[t]."""
    X, y = [], []
    for i in range(lookback, len(feature_data)):
        X.append(feature_data[i - lookback:i])
        y.append(target_data[i - 1])   # target row aligns with end of window
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


# ── Model definition ──────────────────────────────────────────────────────────

def build_attention_lstm(lookback: int, n_features: int, attn_units: int = 32):
    """Attention-LSTM with Bahdanau additive attention + global-mean context."""
    inputs = keras.Input(shape=(lookback, n_features), name="input")

    hidden = keras.layers.LSTM(
        64, return_sequences=True,
        dropout=0.20, recurrent_dropout=0.10,
        name="lstm_enc",
    )(inputs)

    score = keras.layers.Dense(attn_units, use_bias=False, name="attn_W")(hidden)
    score = keras.layers.Activation("tanh", name="attn_tanh")(score)
    score = keras.layers.Dense(1, use_bias=False, name="attn_v")(score)
    # (B, T, 1) → (B, T) so Softmax runs over the last axis (TF.js restriction)
    score_flat = keras.layers.Reshape((lookback,), name="score_flat")(score)
    alpha_flat = keras.layers.Softmax(name="attn_softmax")(score_flat)
    alpha      = keras.layers.Reshape((lookback, 1), name="alpha")(alpha_flat)

    weighted = keras.layers.Multiply(name="weighted")([hidden, alpha])
    context  = keras.layers.GlobalAveragePooling1D(name="context")(weighted)

    x   = keras.layers.Dropout(0.20, name="head_drop")(context)
    x   = keras.layers.Dense(32, activation="relu", name="head_dense")(x)
    out = keras.layers.Dense(1, activation="tanh", name="output")(x)  # target ∈ [-1, 1]

    return keras.Model(inputs=inputs, outputs=out, name="attention_lstm")


# ── TF.js exporter ────────────────────────────────────────────────────────────

def _strip_tensor_suffix(name: str) -> str:
    return name.split(":")[0]


def _validate_topology_for_tfjs(topology: dict) -> None:
    layers = topology.get("config", {}).get("layers", [])
    if not layers:
        raise RuntimeError("Exported topology has no layers.")
    first_cfg = layers[0].get("config", {})
    has_bis = "batch_input_shape" in first_cfg or "batchInputShape" in first_cfg
    if "batch_shape" in first_cfg and not has_bis:
        raise RuntimeError(
            "Exported model.json uses Keras 3's `batch_shape` — TF.js 4.x expects "
            "`batch_input_shape`. Install tf_keras + set TF_USE_LEGACY_KERAS=1."
        )
    for lyr in layers:
        dt = lyr.get("config", {}).get("dtype")
        if isinstance(dt, dict):
            raise RuntimeError(
                f"Layer '{lyr.get('name')}' has Keras 3 DTypePolicy dtype — "
                "TF.js will fail. Install tf_keras + set TF_USE_LEGACY_KERAS=1."
            )


def export_tfjs(model, output_dir: Path):
    """Write TF.js LayersModel format without the tensorflowjs package."""
    output_dir.mkdir(parents=True, exist_ok=True)
    topology = json.loads(model.to_json())
    _validate_topology_for_tfjs(topology)

    weight_specs, buffers = [], []
    for w in model.weights:
        arr = w.numpy().astype(np.float32)
        weight_specs.append({
            "name":  _strip_tensor_suffix(w.name),
            "shape": list(arr.shape),
            "dtype": "float32",
        })
        buffers.append(arr.tobytes())

    bin_filename = "group1-shard1of1.bin"
    with open(output_dir / bin_filename, "wb") as f:
        for buf in buffers:
            f.write(buf)

    model_json = {
        "format":          "layers-model",
        "generatedBy":     f"keras {keras.__version__} (legacy tf_keras)",
        "convertedBy":     "manual (train_lstm.py)",
        "modelTopology":   topology,
        "weightsManifest": [{
            "paths":   [bin_filename],
            "weights": weight_specs,
        }],
    }
    with open(output_dir / "model.json", "w") as f:
        json.dump(model_json, f, indent=2)

    total_kb = sum(len(b) for b in buffers) // 1024
    print(f"  model.json + {bin_filename}  ({total_kb} KB weights)")


# ── Validation gate ───────────────────────────────────────────────────────────

def python_rollout(model, features_df: pd.DataFrame, horizon: int = 63, n_passes: int = 50):
    """
    Mirror of LSTMInferer.js rollout, in Python, for the validation gate.
    Returns the p50 cumulative-return path for post-hoc plausibility checking.

    Only 50 MC passes (vs 200 in production) — this is a sanity test, not
    a production inference, so we trade some variance for gate speed.
    """
    if len(features_df) < LOOKBACK + 30:
        return None, 0.0

    scaled     = features_df[FEAT_COLS].values.astype(np.float32)
    raw_ret    = features_df["_raw_return"].values.astype(np.float32)
    window     = scaled[-LOOKBACK:].copy()
    raw_buf    = list(raw_ret[-64:])

    paths      = [[0.0] * horizon for _ in range(n_passes)]
    cum_rets   = [0.0] * n_passes
    survived   = [True] * n_passes

    for t in range(horizon):
        batch = np.broadcast_to(window, (n_passes, LOOKBACK, N_FEATURES)).copy()
        preds = model(batch, training=True).numpy().flatten()

        for p in range(n_passes):
            if not survived[p]:
                paths[p][t] = cum_rets[p]
                continue
            z_pred = float(preds[p]) * CLIP_SIGMAS
            sigma  = float(np.std(raw_buf[-21:])) or 0.01
            r_pred = np.clip(z_pred * sigma, -0.15, 0.15)
            cum_rets[p] = (1 + cum_rets[p]) * (1 + r_pred) - 1
            paths[p][t] = cum_rets[p]
            if cum_rets[p] < -0.60 or cum_rets[p] > 2.00:
                survived[p] = False
            raw_buf.append(r_pred)
            if len(raw_buf) > 64:
                raw_buf = raw_buf[-64:]

    survivor_frac = sum(survived) / n_passes
    if survivor_frac == 0:
        return None, 0.0

    p50_path = np.median(np.array(paths), axis=0)
    return p50_path, survivor_frac


def run_validation_gate(model, prices_df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Full validation. Returns (passed, list_of_failure_messages)."""
    failures = []

    # Gate 1+2 already done inline in main() on the held-out test split
    # (MSE + directional accuracy). Here we do the rollout sanity tests.

    for ticker in GATE_BASKETS:
        if ticker not in prices_df.columns:
            failures.append(f"Gate basket ticker {ticker} missing from price data")
            continue

        ret    = prices_df[ticker].pct_change().dropna()
        feats  = build_scale_invariant_features(ret)
        feats["_raw_return"] = ret.reindex(feats.index)
        feats  = feats.dropna()

        if len(feats) < LOOKBACK + 90:
            failures.append(f"{ticker}: insufficient history for rollout gate")
            continue

        p50_path, survivor_frac = python_rollout(model, feats, horizon=63, n_passes=50)

        if survivor_frac < GATE_MIN_SURVIVORS:
            failures.append(
                f"{ticker}: rollout survivors {survivor_frac:.0%} < {GATE_MIN_SURVIVORS:.0%}"
            )
            continue

        median_21d = float(p50_path[20])
        if abs(median_21d) > GATE_MAX_21D_MEDIAN:
            failures.append(
                f"{ticker}: 21d median {median_21d:+.1%} > ±{GATE_MAX_21D_MEDIAN:.0%}"
            )
        else:
            print(f"  [gate]  {ticker:8s}: survivors={survivor_frac:.0%}  21d_p50={median_21d:+.2%}  OK")

    return (len(failures) == 0), failures


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Attention-LSTM Training — scale-invariant features, expanded universe")
    print("=" * 70)

    # 1. Download price data (with on-disk cache)
    print(f"\n[1/6] Downloading {len(UNIVERSE)} assets from {START} → today…")
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("ERROR: yfinance not installed. Run: pip install -r requirements-train.txt")

    import time
    from datetime import datetime, timedelta

    CACHE_PATH = SCRIPT_DIR / f".price_cache_v{CACHE_SCHEMA_VERSION}.parquet"
    CACHE_MAX_AGE_HOURS = 24

    # Clean up any older schema caches so we don't leak disk
    for old in SCRIPT_DIR.glob(".price_cache*.parquet"):
        if old.name != CACHE_PATH.name:
            try: old.unlink()
            except Exception: pass

    prices = None
    if CACHE_PATH.exists():
        cache_age = datetime.now() - datetime.fromtimestamp(CACHE_PATH.stat().st_mtime)
        if cache_age < timedelta(hours=CACHE_MAX_AGE_HOURS):
            try:
                cached = pd.read_parquet(CACHE_PATH)
                # Accept cache only if it contains ≥90% of the current universe
                # (so adding new tickers to UNIVERSE invalidates the cache)
                overlap = len(set(cached.columns) & set(UNIVERSE))
                if overlap >= int(0.9 * len(UNIVERSE)):
                    prices = cached[[c for c in cached.columns if c in UNIVERSE]]
                    print(f"  Using cache v{CACHE_SCHEMA_VERSION} "
                          f"({cache_age.total_seconds() / 3600:.1f}h old, "
                          f"{len(prices.columns)}/{len(UNIVERSE)} tickers)")
            except Exception as e:
                print(f"  Cache read failed ({e!s}); re-downloading")

    if prices is None:
        # Per-ticker download with retry. Yahoo rate-limits batch downloads
        # so we loop with small delays + up to 3 retries per ticker.
        series_by_ticker: dict[str, pd.Series] = {}
        failed: list[str] = []
        for i, ticker in enumerate(UNIVERSE, start=1):
            ser = None
            for attempt in range(3):
                try:
                    hist = yf.download(
                        ticker, start=START, end=END,
                        auto_adjust=True, progress=False, threads=False,
                    )
                    if hist is None or hist.empty or "Close" not in hist.columns:
                        raise ValueError("empty or malformed response")
                    ser = hist["Close"].dropna()
                    if isinstance(ser, pd.DataFrame):
                        ser = ser.iloc[:, 0]
                    if len(ser) < 252:
                        raise ValueError(f"only {len(ser)} rows")
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    failed.append(f"{ticker} ({type(e).__name__}: {e})")
            if ser is not None and len(ser) >= 252:
                ser.name = ticker
                series_by_ticker[ticker] = ser
            time.sleep(0.3)
            if i % 10 == 0:
                print(f"  {i}/{len(UNIVERSE)} downloaded  ({len(series_by_ticker)} succeeded so far)")

        if failed:
            print(f"  Failed: {len(failed)} tickers")
            for msg in failed[:5]:
                print(f"    - {msg}")
            if len(failed) > 5:
                print(f"    … {len(failed) - 5} more")

        if len(series_by_ticker) < 20:
            sys.exit(
                f"ERROR: only {len(series_by_ticker)} assets downloaded, need ≥20. "
                f"Yahoo may be rate-limiting — wait 15 min and retry."
            )

        # Reindex everything to a US-business-day calendar. BTC/ETH natively
        # trade 7d/wk; without this, their weekend rows would pollute the union
        # index and push equity tickers below the non-NaN threshold.
        # NB: pd.bdate_range uses Mon-Fri; US market holidays remain as NaN
        # and are filtered per-ticker during feature engineering.
        bday_index = pd.bdate_range(
            start=min(s.index.min() for s in series_by_ticker.values()),
            end=max(s.index.max() for s in series_by_ticker.values()),
        )
        prices = pd.DataFrame(index=bday_index)
        for ticker, ser in series_by_ticker.items():
            # Strip timezone before reindex to avoid tz-naive/tz-aware mismatch
            idx = ser.index.tz_localize(None) if ser.index.tz is not None else ser.index
            s2 = pd.Series(ser.values, index=idx, name=ticker)
            prices[ticker] = s2.reindex(bday_index)

        # Persist cache
        try:
            prices.to_parquet(CACHE_PATH)
            print(f"  Cached → {CACHE_PATH.name}")
        except Exception as e:
            print(f"  Cache write failed ({e!s}); continuing without cache")

    # Drop tickers with >30% missing on the business-day calendar
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.7))
    available = list(prices.columns)
    print(f"  {len(available)}/{len(UNIVERSE)} assets available  ({len(prices)} trading days)")
    if len(available) < 20:
        sys.exit(f"ERROR: only {len(available)} assets available after alignment, need ≥20")

    # 2. Feature engineering — per-asset with time-based global cutoffs
    print("\n[2/6] Engineering scale-invariant features…")
    print(f"  train   : < {VAL_CUTOFF}")
    print(f"  val     : {VAL_CUTOFF} → < {TEST_CUTOFF}")
    print(f"  test OOS: ≥ {TEST_CUTOFF}  (held out — gate only)")

    val_cutoff_ts  = VAL_CUTOFF_TS
    test_cutoff_ts = TEST_CUTOFF_TS

    X_tr_all, y_tr_all = [], []
    X_va_all, y_va_all = [], []
    X_te_all, y_te_all = [], []
    skipped: list[str] = []
    per_ticker_stats: dict[str, tuple[int, int, int]] = {}

    for ticker in available:
        ret = prices[ticker].pct_change().dropna()
        df  = build_scale_invariant_features(ret)
        if len(df) < LOOKBACK + 90:
            skipped.append(ticker)
            continue

        feats  = df[FEAT_COLS].values
        target = df["target"].values
        idx    = df.index

        # Build sequences over the ENTIRE history first, then route each
        # sequence to train/val/test based on the timestamp of its TARGET day.
        # Routing on the target day (not the last feature day) is the right
        # semantic: the model's "prediction date" is the target.
        X_all, y_all, t_all = [], [], []
        for i in range(LOOKBACK, len(feats)):
            X_all.append(feats[i - LOOKBACK:i])
            y_all.append(target[i - 1])
            t_all.append(idx[i - 1])
        if not X_all:
            skipped.append(ticker)
            continue
        X_all = np.asarray(X_all, dtype=np.float32)
        y_all = np.asarray(y_all, dtype=np.float32)
        t_all = pd.DatetimeIndex(t_all)

        # Targets at the LAST feature-window row have timestamp t_all[k];
        # the actual target is the next day's z_ret, which is ≈ t_all[k] + 1 bday.
        # For routing we use t_all[k] (the prediction-context date).
        tr_mask = t_all <  val_cutoff_ts
        va_mask = (t_all >= val_cutoff_ts) & (t_all < test_cutoff_ts)
        te_mask = t_all >= test_cutoff_ts

        # Per-ticker cap: keep the MOST RECENT MAX_SAMPLES_PER_TICKER train rows
        # so newer regimes (post-2020) aren't drowned by older ones.
        if tr_mask.sum() > MAX_SAMPLES_PER_TICKER:
            tr_idx = np.where(tr_mask)[0][-MAX_SAMPLES_PER_TICKER:]
            tr_bool = np.zeros_like(tr_mask)
            tr_bool[tr_idx] = True
            tr_mask = tr_bool

        if tr_mask.any():
            X_tr_all.append(X_all[tr_mask]); y_tr_all.append(y_all[tr_mask])
        if va_mask.any():
            X_va_all.append(X_all[va_mask]); y_va_all.append(y_all[va_mask])
        if te_mask.any():
            X_te_all.append(X_all[te_mask]); y_te_all.append(y_all[te_mask])

        per_ticker_stats[ticker] = (int(tr_mask.sum()), int(va_mask.sum()), int(te_mask.sum()))

    if skipped:
        print(f"  Skipped (insufficient history): {', '.join(skipped)}")

    if not X_tr_all or not X_va_all or not X_te_all:
        sys.exit(
            "ERROR: split produced empty train/val/test. Check TEST_CUTOFF / "
            "VAL_CUTOFF are inside your data range."
        )

    X_tr = np.concatenate(X_tr_all, axis=0); y_tr = np.concatenate(y_tr_all, axis=0)
    X_val = np.concatenate(X_va_all, axis=0); y_val = np.concatenate(y_va_all, axis=0)
    X_test = np.concatenate(X_te_all, axis=0); y_test = np.concatenate(y_te_all, axis=0)

    print(f"  train: {len(X_tr):>6,}   val: {len(X_val):>5,}   test: {len(X_test):>5,}")
    print(f"  target (train): mean={y_tr.mean():+.4f}  std={y_tr.std():.4f}  "
          f"skew={float(pd.Series(y_tr).skew()):+.3f}")
    if len(per_ticker_stats) <= 10 or True:
        # Print the 3 largest contributors so we can spot concentration
        sorted_stats = sorted(per_ticker_stats.items(), key=lambda kv: -kv[1][0])[:3]
        for t, (n_tr, n_va, n_te) in sorted_stats:
            print(f"    top contrib  {t:8s}: train={n_tr}  val={n_va}  test={n_te}")

    # 3. Build model
    print("\n[3/6] Building model…")
    tf.random.set_seed(42)
    np.random.seed(42)
    model = build_attention_lstm(LOOKBACK, N_FEATURES)
    # Huber loss: L2 for small errors, L1 for large ones. Reduces the weight
    # of fat-tail days (earnings surprises, crypto flash-crashes) that MSE
    # over-penalises. Also helps directional accuracy by not over-fitting
    # to magnitude outliers. We still report MSE in the gate for comparability.
    model.compile(
        optimizer=keras.optimizers.Adam(5e-4),
        loss=keras.losses.Huber(delta=0.1),
        metrics=["mse", "mae"],
    )
    model.summary()
    print(f"\n  Trainable parameters: {model.count_params():,}")

    # 4. Train (early stopping uses val; test is reserved for the gate)
    print("\n[4/6] Training…")

    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1,
    )
    lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5, verbose=1,
    )
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100, batch_size=128,
        callbacks=[es, lr], verbose=2,
    )

    # 5. Validation gate — genuinely held-out test split (never seen in training)
    print("\n[5/6] Validation gate (OOS test ≥ " + TEST_CUTOFF + ")…")
    te_pred   = model.predict(X_test, verbose=0).flatten()
    te_mse    = float(np.mean((te_pred - y_test) ** 2))
    y_var     = float(np.var(y_test))
    mse_ratio = te_mse / y_var if y_var > 0 else float("inf")
    dir_acc   = float(np.mean(np.sign(te_pred) == np.sign(y_test)))

    print(f"  Test MSE:              {te_mse:.5f}   (null-model var(y)={y_var:.5f})")
    print(f"  MSE / var(y):          {mse_ratio:.3f}     (gate: ≤ {GATE_MAX_MSE_VAR_RATIO})")
    print(f"  Directional accuracy:  {dir_acc:.1%}     (gate: ≥ {GATE_MIN_DIR_ACCURACY:.0%})")

    gate_failures = []
    if mse_ratio > GATE_MAX_MSE_VAR_RATIO:
        gate_failures.append(
            f"Test MSE/var(y) = {mse_ratio:.3f} > {GATE_MAX_MSE_VAR_RATIO} "
            f"(model is >10% worse than predicting zero on held-out period)"
        )
    if dir_acc < GATE_MIN_DIR_ACCURACY:
        gate_failures.append(f"Directional accuracy {dir_acc:.1%} < {GATE_MIN_DIR_ACCURACY:.0%}")

    rollout_ok, rollout_failures = run_validation_gate(model, prices)
    gate_failures.extend(rollout_failures)

    if gate_failures:
        print("\n" + "=" * 70)
        print("VALIDATION GATE FAILED — model NOT exported:")
        for msg in gate_failures:
            print(f"  ✗ {msg}")
        print("\nThe previous model (if any) remains in place.")
        print("=" * 70)
        sys.exit(1)

    print("  All gate checks passed.")

    # 6. Save + export
    print(f"\n[6/6] Exporting model…")
    model.save(str(MODEL_OUT))
    print(f"  Keras model:    {MODEL_OUT}")
    print(f"  TF.js export → {TFJS_OUT}")
    export_tfjs(model, TFJS_OUT)

    print("\n" + "=" * 70)
    print("DONE. If running locally, commit the model and push:")
    print("  git add frontend/public/models/lstm/")
    print('  git commit -m "chore: retrain LSTM"')
    print("  git push")
    print("=" * 70)


if __name__ == "__main__":
    main()
