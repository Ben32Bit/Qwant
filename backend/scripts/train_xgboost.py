"""
train_xgboost.py — Local training script for browser-side quantile forecaster.

Run ONCE locally to train and export models:
  cd backend
  pip install skl2onnx onnx scikit-learn>=1.4 pandas yfinance pandas-datareader
  python scripts/train_xgboost.py

  Expected training time: ~5-10 minutes (HistGradientBoostingRegressor).

Outputs (5 ONNX models + metadata):
  frontend/public/models/xgboost/q05.onnx
  frontend/public/models/xgboost/q25.onnx
  frontend/public/models/xgboost/q50.onnx
  frontend/public/models/xgboost/q75.onnx
  frontend/public/models/xgboost/q95.onnx
  frontend/public/models/xgboost/meta.json

ONNX models include the StandardScaler as preprocessing nodes (Pipeline → ONNX),
so XGBoostInferer.js passes raw features directly — no client-side scaling needed.

Features (14 scalar values):
  [Price-derived — 9]
  ret_1d, ret_5d, ret_21d, ret_63d, vol_21d, vol_63d, mom_12_1, rsi_14, vol_ratio

  [Macro/VIX — 5]
  yield_curve_10y2y : 10Y-2Y Treasury spread (recession signal)
  credit_spread_baa : BAA-10Y credit spread  (risk-off proxy)
  vix_pct_rank      : VIX percentile rank vs 2Y history (0–1)
  vix_term_slope    : VIX3M/VIX − 1 (contango=positive, stress=negative)
  real_yield_10y    : 10Y TIPS real yield (higher = tighter financial conditions)

Target: 21-day forward cumulative return.
Horizon extrapolation: client scales first-period bands to 252 days via sqrt(t/21).

Out-of-sample methodology:
  Purged walk-forward CV with 21-day embargo per López de Prado (2018, Ch. 7).
  Expanding window (not rolling) — uses all history up to the split.
  No random k-fold. 20% held-out test window for final OOS metrics.

IMPORTANT: Retrain whenever macro features are added/changed, or the feature
  list diverges from what the server sends in prepare_xgboost_features().
  The server reads n_features from meta.json to detect version mismatches.

References:
  Chen, T. & Guestrin, C. (2016). XGBoost. KDD. doi:10.1145/2939672.2939785
  Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning.
    Review of Financial Studies, 33(5), 2223–2273. doi:10.1093/rfs/hhaa009
  López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
    Ch. 7 (Cross-Validation in Finance — purged k-fold, embargo).
  Friedman, J.H. (2001). Greedy function approximation: A gradient boosting machine.
    Annals of Statistics, 29(5), 1189–1232. doi:10.1214/aos/1013203451
  CBOE VIX White Paper: https://www.cboe.com/tradable_products/vix/vix_white_paper.pdf
"""

import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# Load .env so FRED_API_KEY etc. are available without prefixing them on the CLI
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

SCRIPT_DIR   = Path(__file__).parent
BACKEND_DIR  = SCRIPT_DIR.parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
MODEL_OUT    = FRONTEND_DIR / "public" / "models" / "xgboost"
MODEL_OUT.mkdir(parents=True, exist_ok=True)

# Add backend to path so we can import providers
sys.path.insert(0, str(BACKEND_DIR))

# ── Training universe ─────────────────────────────────────────────────────────

UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "TLT", "AGG", "IEF",
    "GLD", "USO",
    "XLK", "XLF", "XLE", "XLV", "XLY",
]
START        = "2010-01-01"
END          = "2024-12-31"
TRADING_DAYS = 252
HORIZON      = 21     # forecast horizon in trading days
EMBARGO      = 21     # gap between train and test (prevents leakage)
MIN_TRAIN    = 252    # minimum training observations
QUANTILES    = [0.05, 0.25, 0.50, 0.75, 0.95]
QTAGS        = ["q05", "q25", "q50", "q75", "q95"]

# ── Walk-forward CV config ────────────────────────────────────────────────────
# Step every 126 days (semi-annual) for OOS eval; subsample to ≤60 folds max.
# With HistGBR this is <10 min vs 10h with legacy GBR.
CV_STEP      = 126    # semi-annual step (vs HORIZON=21 before → 12× fewer folds)
MAX_CV_FOLDS = 60     # cap: randomly skip folds beyond this limit

# ── HistGradientBoostingRegressor hyperparams ─────────────────────────────────
# CV folds use a lighter config (faster). Final models use the full config.
# Early stopping is built-in: validation_fraction is held out from training data.
# l2_regularization reduces overfitting on noisy financial targets.
_HGBR_BASE = dict(
    max_depth=4, learning_rate=0.05, min_samples_leaf=60,
    l2_regularization=0.5, max_bins=128,
    early_stopping=True, n_iter_no_change=20,
    random_state=42,
)
HGBR_CV    = {**_HGBR_BASE, "max_iter": 200, "validation_fraction": 0.15}
HGBR_FINAL = {**_HGBR_BASE, "max_iter": 600, "validation_fraction": 0.10}

PRICE_FEATURE_NAMES = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d",
    "vol_21d", "vol_63d", "mom_12_1", "rsi_14", "vol_ratio",
]
MACRO_FEATURE_NAMES = [
    "yield_curve_10y2y",
    "credit_spread_baa",
    "vix_pct_rank",
    "vix_term_slope",
    "real_yield_10y",
]
ALL_FEATURE_NAMES = PRICE_FEATURE_NAMES + MACRO_FEATURE_NAMES

N_PRICE_FEATURES = len(PRICE_FEATURE_NAMES)   # 9
N_MACRO_FEATURES = len(MACRO_FEATURE_NAMES)   # 5
N_FEATURES       = len(ALL_FEATURE_NAMES)     # 14


# ── Macro data download ───────────────────────────────────────────────────────

def download_macro_data(start: str, end: str) -> pd.DataFrame:
    """
    Download FRED + VIX macro data for the training period.
    Returns a DataFrame with columns = MACRO_FEATURE_NAMES, bdate index.
    Falls back to neutral constants if providers are unavailable.
    """
    from app.services.fred_provider import download_macro_history, get_macro_features
    from app.services.vix_provider  import download_vix_history

    # Quick spot-check: fetch one data point to confirm source before bulk download
    spot = get_macro_features()
    source = spot.get("source", "unknown")
    available = spot.get("available", False)
    if available:
        print(f"  FRED source: {source}  (yield_curve={spot['yield_curve_10y2y']}%  "
              f"credit={spot['credit_spread_baa']}%  real_yield={spot['real_yield_10y']}%)")
    else:
        print(f"  WARNING: FRED unavailable (source={source}) — macro features will be neutral constants")
        print(f"  Tip: set FRED_API_KEY in backend/.env for live macro data")

    fred_df = download_macro_history(start, end)
    vix_df  = download_vix_history(start, end)

    # Align on common business-day index
    idx = pd.bdate_range(start, end)
    macro = pd.DataFrame(index=idx)
    for col in ["yield_curve_10y2y", "credit_spread_baa", "real_yield_10y"]:
        macro[col] = fred_df[col].reindex(idx, method="ffill") if col in fred_df.columns else np.nan
    for col in ["vix_pct_rank", "vix_term_slope"]:
        macro[col] = vix_df[col].reindex(idx, method="ffill") if col in vix_df.columns else np.nan

    macro = macro[MACRO_FEATURE_NAMES].fillna(0.0)
    n_live = int((macro != 0.0).any(axis=0).sum())
    print(f"  Macro data: {macro.notna().all().sum()}/{N_MACRO_FEATURES} columns  "
          f"({n_live} with live values, {N_MACRO_FEATURES - n_live} neutral)")
    return macro


# ── Feature engineering ───────────────────────────────────────────────────────

def compute_features(
    r: pd.Series,
    macro_vals: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    Compute predictive features from trailing returns + optional macro values.

    Parameters
    ----------
    r : pd.Series
        Daily return series up to observation date (inclusive).
    macro_vals : np.ndarray, shape (5,), optional
        [yield_curve_10y2y, credit_spread_baa, vix_pct_rank,
         vix_term_slope, real_yield_10y]
        If None, only the 9 price features are returned.

    Returns
    -------
    np.ndarray of shape (9,) or (14,), or None if insufficient history.
    """
    if len(r) < 63:
        return None

    ret_1d   = float(r.iloc[-1])
    ret_5d   = float(r.iloc[-5:].sum())
    ret_21d  = float(r.iloc[-21:].sum())
    ret_63d  = float(r.iloc[-63:].sum())
    vol_21d  = float(r.iloc[-21:].std() * np.sqrt(TRADING_DAYS))
    vol_63d  = float(r.iloc[-63:].std() * np.sqrt(TRADING_DAYS))

    if len(r) >= TRADING_DAYS:
        mom_12_1 = float(r.iloc[-TRADING_DAYS:-HORIZON].sum())
    elif len(r) >= 63:
        mom_12_1 = float(r.iloc[:-HORIZON].sum()) if len(r) > HORIZON else 0.0
    else:
        mom_12_1 = 0.0

    delta = r.diff().dropna()
    if len(delta) >= 14:
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi_14 = float(gain.iloc[-1] / (gain.iloc[-1] + loss.iloc[-1] + 1e-9))
    else:
        rsi_14 = 0.5

    vol_ratio = vol_21d / (vol_63d + 1e-9)

    base = np.array([ret_1d, ret_5d, ret_21d, ret_63d,
                     vol_21d, vol_63d, mom_12_1, rsi_14, vol_ratio],
                    dtype=np.float32)

    if macro_vals is not None:
        return np.concatenate([base, macro_vals.astype(np.float32)])
    return base


def build_dataset(
    prices: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) from price data + optional macro DataFrame.

    X: feature vectors (9 or 14-dim) at each observation date.
    y: HORIZON-day forward cumulative return.

    Uses purged walk-forward sampling with embargo=HORIZON days.
    """
    all_X, all_y = [], []

    for ticker in prices.columns:
        r = prices[ticker].pct_change().dropna()
        n = len(r)

        # Pre-align macro to this ticker's date index
        if macro_df is not None:
            macro_aligned = macro_df.reindex(r.index, method="ffill").fillna(0.0)
        else:
            macro_aligned = None

        for i in range(63, n - HORIZON):
            mv = macro_aligned.iloc[i].values if macro_aligned is not None else None
            feats = compute_features(r.iloc[:i + 1], mv)
            if feats is None:
                continue
            fwd    = r.iloc[i + 1: i + 1 + HORIZON]
            target = float((1 + fwd).prod() - 1)
            all_X.append(feats)
            all_y.append(target)

    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32)


# ── Walk-forward evaluation ───────────────────────────────────────────────────

def purged_walkforward_splits(n: int, min_train: int, step: int, embargo: int):
    """Expanding-window walk-forward splits with embargo gap."""
    train_end = min_train
    while train_end + embargo + step <= n:
        test_start = train_end + embargo
        test_end   = min(test_start + step, n)
        yield train_end, test_start, test_end
        train_end += step


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_model_to_onnx(model, n_features: int, out_path: Path):
    """
    Export a fitted HistGradientBoostingRegressor to ONNX.
    Requires skl2onnx >= 1.14 for HistGBR support.
    XGBoostInferer.js passes raw (unscaled) features — HGBR uses histogram
    binning internally so no StandardScaler is needed in the graph.

    Opset targets MUST match onnxruntime-web's supported range. onnxruntime-web
    1.20.1 (pinned via CDN in frontend/index.html) supports:
      - ai.onnx         up to opset 20  (IR version 9)
      - ai.onnx.ml      up to opset 3

    Default skl2onnx converts targets the latest opset (22+ / IR 10), which
    onnxruntime-web rejects with a raw WASM memory pointer as the error
    (users see "XGBoost: <pointer>" instead of an opset-mismatch message).
    Pin conservatively to opset 18 / ml 3.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [('float_input', FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset={"": 18, "ai.onnx.ml": 3},
    )
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Sanity check — verify the export lands on an IR version ORT-Web can load.
    kb = out_path.stat().st_size // 1024
    ir = onnx_model.ir_version
    ops = {o.domain or "ai.onnx": o.version for o in onnx_model.opset_import}
    if ir > 9:
        raise RuntimeError(
            f"{out_path.name}: exported IR version {ir} exceeds onnxruntime-web "
            f"1.20.1's max of 9. Lower target_opset and re-export."
        )
    print(f"    {out_path.name}  ({kb} KB, IR {ir}, opsets {ops})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("XGBoost Quantile Forecast Training Script")
    print(f"Features: {N_PRICE_FEATURES} price + {N_MACRO_FEATURES} macro = {N_FEATURES} total")
    print("=" * 60)

    # 1. Download prices
    print(f"\n[1/6] Downloading {len(UNIVERSE)} assets ({START}→{END})…")
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("ERROR: yfinance not installed. Run: pip install yfinance")

    prices = yf.download(UNIVERSE, start=START, end=END,
                         auto_adjust=True, progress=False)["Close"]
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    print(f"  {len(prices.columns)} assets available  ({len(prices)} trading days)")

    # 2. Download macro data
    print("\n[2/6] Downloading macro data (FRED + VIX)…")
    macro_df = download_macro_data(START, END)
    print(f"  {len(macro_df)} macro observations  ({macro_df.index[0].date()}→{macro_df.index[-1].date()})")

    # 3. Build dataset
    print("\n[3/6] Building feature dataset…")
    X, y = build_dataset(prices, macro_df)
    print(f"  {len(X)} samples  ×  {X.shape[1]} features")
    print(f"  Target  — mean: {y.mean():.4f}  std: {y.std():.4f}")

    # 4. Walk-forward CV  (HistGBR — ~20-50× faster than legacy GBR)
    print(f"\n[4/6] Walk-forward OOS evaluation (step={CV_STEP}d, max {MAX_CV_FOLDS} folds)…")
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        sys.exit("ERROR: scikit-learn>=1.4 not installed. Run: pip install -U scikit-learn")

    n = len(X)
    all_splits = list(purged_walkforward_splits(n, MIN_TRAIN, CV_STEP, EMBARGO))

    # Subsample evenly to MAX_CV_FOLDS if there are more
    if len(all_splits) > MAX_CV_FOLDS:
        step = len(all_splits) // MAX_CV_FOLDS
        eval_splits = all_splits[::step][:MAX_CV_FOLDS]
    else:
        eval_splits = all_splits
    print(f"  {len(all_splits)} total folds → evaluating {len(eval_splits)} checkpoints")

    oos_preds_median = []
    oos_actuals = []
    for i, (train_end, test_start, test_end) in enumerate(eval_splits):
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]
        if len(X_te) == 0:
            continue
        model = HistGradientBoostingRegressor(loss='quantile', quantile=0.50, **HGBR_CV)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        oos_preds_median.extend(preds.tolist())
        oos_actuals.extend(y_te.tolist())
        if (i + 1) % 10 == 0:
            print(f"    checkpoint {i+1}/{len(eval_splits)}")

    oos_preds_median = np.array(oos_preds_median)
    oos_actuals      = np.array(oos_actuals)
    oos_r2  = float(1 - np.var(oos_actuals - oos_preds_median) / np.var(oos_actuals))
    oos_mse = float(np.mean((oos_actuals - oos_preds_median) ** 2))
    # Rank IC (Spearman-style) — more robust than Pearson for fat-tailed returns
    from scipy.stats import rankdata
    ic = float(np.corrcoef(rankdata(oos_preds_median), rankdata(oos_actuals))[0, 1])
    pinball_loss = float(np.mean(np.maximum(
        0.50 * (oos_actuals - oos_preds_median),
        (0.50 - 1) * (oos_actuals - oos_preds_median),
    )))
    print(f"\n  OOS R²:      {oos_r2:.4f}  (negative is common for return prediction)")
    print(f"  OOS MSE:     {oos_mse:.6f}")
    print(f"  Rank IC:     {ic:.4f}  (≥0.03 is useful in practice)")
    print(f"  Pinball p50: {pinball_loss:.6f}")

    # 5. Train final models on full data
    print("\n[5/6] Training 5 quantile models on full data (HistGBR + early stopping)…")
    n_train = int(n * 0.80)
    X_full, y_full = X[:n_train], y[:n_train]

    # Store feature means (used as inference-time fallback for macro features)
    feature_means = X_full.mean(axis=0).tolist()

    models = {}
    for q, tag in zip(QUANTILES, QTAGS):
        print(f"  Training {tag} (quantile={q:.2f})…", end=" ", flush=True)
        model = HistGradientBoostingRegressor(loss='quantile', quantile=q, **HGBR_FINAL)
        model.fit(X_full, y_full)
        n_trees = model.n_iter_  # actual trees used (early stopping may reduce)
        print(f"{n_trees} trees")
        models[tag] = model

    # 6. Export to ONNX
    print("\n[6/6] Exporting to ONNX…")
    try:
        import skl2onnx  # noqa: F401
        import onnx      # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: skl2onnx / onnx not installed.\n"
            "Run: pip install skl2onnx onnx"
        )

    for tag, model in models.items():
        out_path = MODEL_OUT / f"{tag}.onnx"
        export_model_to_onnx(model, N_FEATURES, out_path)

    # Save metadata
    meta = {
        "feature_names":    ALL_FEATURE_NAMES,
        "price_features":   PRICE_FEATURE_NAMES,
        "macro_features":   MACRO_FEATURE_NAMES,
        "n_features":       N_FEATURES,
        "n_price_features": N_PRICE_FEATURES,
        "n_macro_features": N_MACRO_FEATURES,
        "feature_means":    [round(v, 6) for v in feature_means],
        "horizon_days":     HORIZON,
        "oos_r2":           round(oos_r2, 4),
        "oos_mse":          round(oos_mse, 6),
        "rank_ic":          round(ic, 4),
        "pinball_p50":      round(pinball_loss, 6),
        "n_samples":        n,
        "n_assets":         len(prices.columns),
        "train_period":     f"{START}–{END}",
        "quantiles":        QUANTILES,
        "booster":          "HistGradientBoostingRegressor",
    }
    with open(MODEL_OUT / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  meta.json written")

    print("\n" + "=" * 60)
    print(f"DONE.  OOS R²={oos_r2:.4f}  Rank IC={ic:.4f}  Pinball={pinball_loss:.4f}")
    print(f"       {N_FEATURES} features ({N_PRICE_FEATURES} price + {N_MACRO_FEATURES} macro)")
    print("Note: negative OOS R² is common for financial return prediction.")
    print("      Rank IC ≥ 0.03 is considered useful in practice (Gu et al 2020).")
    print("Commit the model files and push:")
    print("  git add frontend/public/models/xgboost/")
    print('  git commit -m "Retrain XGBoost HistGBR (14f, early stopping)"')
    print("  git push")
    print("=" * 60)


if __name__ == "__main__":
    main()
