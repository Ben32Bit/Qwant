"""
train_lstm.py — Local training script for the browser-side Attention-LSTM.

Run ONCE locally to train and export the model:
  cd backend
  pip install tensorflow tf_keras scikit-learn pandas yfinance
  python scripts/train_lstm.py

Outputs:
  backend/scripts/lstm_model.keras           (Keras saved model)
  frontend/public/models/lstm/model.json     (TF.js model config — Keras 2 format)
  frontend/public/models/lstm/group1-shard1of1.bin  (weight data)

IMPORTANT — Keras 2 vs Keras 3:
  TF.js 4.x LayersModel loader only reads the legacy Keras 2 JSON format
  (batch_input_shape, flat dtype strings, layer-scoped weight names).
  TF 2.16+ ships Keras 3 by default, whose JSON uses `batch_shape`,
  `DTypePolicy` objects, and bare weight names — this causes
  TF.js to fail with "InputLayer should be passed either batchInputShape
  or inputShape".

  This script forces legacy Keras 2 via the `tf_keras` package and the
  TF_USE_LEGACY_KERAS env var. Make sure tf_keras is installed:
      pip install tf_keras

Architecture: Input(60,5) → LSTM(64) → Bahdanau attention
              → GlobalAveragePooling1D → Dropout(0.20) → Dense(32) → Dense(1)
Model size: ~300-500 KB
"""

import os
# Must be set BEFORE importing tensorflow so TF dispatches to tf_keras.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import tensorflow as tf
except ImportError:
    sys.exit("ERROR: TensorFlow not installed. Run: pip install tensorflow tf_keras")

# Force Keras 2 (legacy) for TF.js compatibility.
try:
    import tf_keras as keras
except ImportError:
    sys.exit(
        "ERROR: tf_keras not installed — required for TF.js-compatible export.\n"
        "Run: pip install tf_keras\n"
        "(TF.js 4.x cannot load Keras 3 JSON. tf_keras provides the Keras 2 API.)"
    )

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
BACKEND_DIR  = SCRIPT_DIR.parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
TFJS_OUT     = FRONTEND_DIR / "public" / "models" / "lstm"
MODEL_OUT    = SCRIPT_DIR / "lstm_model.keras"

TFJS_OUT.mkdir(parents=True, exist_ok=True)

# ── Training universe ─────────────────────────────────────────────────────────
UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "TLT", "AGG", "IEF",
    "GLD", "USO",
    "XLK", "XLF", "XLE", "XLV", "XLY",
]
START        = "2010-01-01"
END          = "2024-12-31"
LOOKBACK     = 60
TRADING_DAYS = 252
FEAT_COLS    = ["r", "vol_21d", "mom_5d", "mom_21d", "rsi_14"]


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(returns: pd.Series) -> pd.DataFrame:
    r  = returns.copy()
    df = pd.DataFrame({"r": r})
    df["vol_21d"] = r.rolling(21).std() * np.sqrt(TRADING_DAYS)
    df["mom_5d"]  = r.rolling(5).sum()
    df["mom_21d"] = r.rolling(21).sum()
    delta         = r.diff()
    gain          = delta.clip(lower=0).rolling(14).mean()
    loss          = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"]  = gain / (gain + loss + 1e-9)
    return df.dropna()


def make_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# ── Model definition ──────────────────────────────────────────────────────────

def build_attention_lstm(lookback: int, n_features: int, attn_units: int = 32):
    """
    Attention-LSTM with Bahdanau additive attention.

    Uses GlobalAveragePooling1D for the context vector aggregation instead
    of tf.reduce_sum / a Lambda layer. GlobalAveragePooling1D is:
      - A standard Keras layer (no custom class, no Lambda closure)
      - Fully pickle-safe (no 'cannot pickle module object' error)
      - Natively supported by TensorFlow.js

    Semantically: since alpha (softmax) already sums to 1 over T, the
    weighted mean == weighted sum / T. The downstream Dense(32) layer
    absorbs the 1/T scale factor during training — no information is lost.

    Architecture:
      Input(lookback, n_features)
      → LSTM(64, return_sequences=True, dropout=0.20, recurrent_dropout=0.10)
      → Bahdanau attention: eₜ = vᵀ·tanh(Wₐ·hₜ), α = softmax(e) over time
      → Multiply([hidden, α])          (B, T, 64)
      → GlobalAveragePooling1D         (B, 64)   ← mean over T
      → Dropout(0.20) → Dense(32, relu) → Dense(1)

    IMPORTANT — TF.js softmax axis restriction:
      TF.js 4.x (kernels/Softmax.ts) throws:
        "Softmax along a non-last dimension is not yet supported"
      when `axis < ndim - 1`. To stay compatible we Reshape (B,T,1) → (B,T),
      softmax over the last (and only) axis, then Reshape back to (B,T,1).
      Mathematically identical; TF.js-safe.
    """
    inputs = keras.Input(shape=(lookback, n_features), name="input")

    hidden = keras.layers.LSTM(
        64, return_sequences=True,
        dropout=0.20, recurrent_dropout=0.10,
        name="lstm_enc",
    )(inputs)

    score  = keras.layers.Dense(attn_units, use_bias=False, name="attn_W")(hidden)
    score  = keras.layers.Activation("tanh", name="attn_tanh")(score)
    score  = keras.layers.Dense(1, use_bias=False, name="attn_v")(score)
    # score shape: (B, T, 1). Reshape to (B, T) so softmax runs on the last axis.
    score_flat = keras.layers.Reshape((lookback,), name="score_flat")(score)
    alpha_flat = keras.layers.Softmax(name="attn_softmax")(score_flat)       # default axis=-1
    alpha      = keras.layers.Reshape((lookback, 1), name="alpha")(alpha_flat)  # back to (B, T, 1)

    # Weighted hidden states; GAP averages over T (equivalent to sum up to 1/T scale)
    weighted = keras.layers.Multiply(name="weighted")([hidden, alpha])
    context  = keras.layers.GlobalAveragePooling1D(name="context")(weighted)

    x   = keras.layers.Dropout(0.20, name="head_drop")(context)
    x   = keras.layers.Dense(32, activation="relu", name="head_dense")(x)
    out = keras.layers.Dense(1, name="output")(x)

    return keras.Model(inputs=inputs, outputs=out, name="attention_lstm")


# ── Manual TF.js exporter ─────────────────────────────────────────────────────
# tensorflowjs_converter CLI is broken with TF 2.16+ (np.object + tensorflow_hub
# compatibility issues). We write the TF.js LayersModel format directly:
#   model.json  — model topology (from model.to_json()) + weights manifest
#   *.bin       — concatenated float32 weight arrays (little-endian)
#
# Requires legacy Keras 2 (tf_keras) — Keras 3's JSON format is not TF.js compatible.

def _strip_tensor_suffix(name: str) -> str:
    # Keras 2 weight names look like "lstm_enc/kernel:0"; TF.js expects the
    # tensor ":0" suffix stripped so the manifest matches what the loader expects.
    return name.split(":")[0]


def _validate_topology_for_tfjs(topology: dict) -> None:
    """Sanity-check the exported model.json is in the format TF.js 4.x expects."""
    layers = topology.get("config", {}).get("layers", [])
    if not layers:
        raise RuntimeError("Exported topology has no layers — model.to_json() returned empty.")

    # Find the first layer with any shape info
    first = layers[0]
    first_cfg = first.get("config", {})
    has_bis = "batch_input_shape" in first_cfg or "batchInputShape" in first_cfg

    # Keras 3 uses "batch_shape"; we need to reject that
    if "batch_shape" in first_cfg and not has_bis:
        raise RuntimeError(
            "Exported model.json uses Keras 3's `batch_shape` field, which TF.js "
            "4.x cannot parse (it wants `batch_input_shape`). Make sure `tf_keras` "
            "is installed and `TF_USE_LEGACY_KERAS=1` is set before importing TF. "
            f"Layer 0 config keys: {sorted(first_cfg.keys())}"
        )

    # Reject Keras 3 DTypePolicy dtype wrappers (TF.js expects flat "float32" strings)
    for lyr in layers:
        dt = lyr.get("config", {}).get("dtype")
        if isinstance(dt, dict):
            raise RuntimeError(
                f"Layer '{lyr.get('name')}' has a Keras 3 DTypePolicy dtype object "
                f"instead of a flat 'float32' string. TF.js will fail to load. "
                "Install tf_keras and set TF_USE_LEGACY_KERAS=1."
            )


def export_tfjs(model, output_dir: Path):
    """Write TF.js LayersModel format without the tensorflowjs package."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model topology from Keras 2 JSON serialisation (tf_keras produces the
    # legacy format natively — no conversion needed).
    topology = json.loads(model.to_json())

    # Fail loudly if the format is not TF.js-compatible
    _validate_topology_for_tfjs(topology)

    # Weight specs + binary payload. TF.js wants layer-scoped names WITHOUT
    # the ":0" TF variable suffix.
    weight_specs = []
    buffers = []
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
        "format":       "layers-model",
        "generatedBy":  f"keras {keras.__version__} (legacy tf_keras)",
        "convertedBy":  "manual (train_lstm.py)",
        "modelTopology": topology,
        "weightsManifest": [{
            "paths":   [bin_filename],
            "weights": weight_specs,
        }],
    }
    with open(output_dir / "model.json", "w") as f:
        json.dump(model_json, f, indent=2)

    total_kb = sum(len(b) for b in buffers) // 1024
    print(f"  model.json + {bin_filename}  ({total_kb} KB weights)")
    print(f"  {len(weight_specs)} weight tensors")
    print(f"  First weight name: {weight_specs[0]['name']}  (should be layer-scoped)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Attention-LSTM Training Script")
    print("=" * 60)

    # 1. Download price data
    print(f"\n[1/5] Downloading {len(UNIVERSE)} assets ({START}→{END})…")
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("ERROR: yfinance not installed. Run: pip install yfinance")

    prices = yf.download(UNIVERSE, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    available = list(prices.columns)
    print(f"  {len(available)} assets available  ({len(prices)} trading days)")

    # 2. Feature engineering
    print("\n[2/5] Engineering features + building sequences…")
    from sklearn.preprocessing import MinMaxScaler

    all_X, all_y = [], []
    for ticker in available:
        ret = prices[ticker].pct_change().dropna()
        df  = build_features(ret)
        if len(df) < LOOKBACK + 90:
            continue
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(df[FEAT_COLS].values)
        n_train = int(len(scaled) * 0.70)
        X_tr, y_tr = make_sequences(scaled[:n_train], LOOKBACK)
        all_X.append(X_tr)
        all_y.append(y_tr)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"  {len(X)} sequences  ×  {X.shape[2]} features")

    # 3. Build model
    print("\n[3/5] Building model…")
    tf.random.set_seed(42)
    np.random.seed(42)
    model = build_attention_lstm(LOOKBACK, len(FEAT_COLS))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.summary()
    print(f"\n  Parameters: {model.count_params():,}")

    # 4. Train
    print("\n[4/5] Training…")
    n_val  = int(len(X) * 0.15)
    X_tr2, X_val = X[:-n_val], X[-n_val:]
    y_tr2, y_val = y[:-n_val], y[-n_val:]

    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                       restore_best_weights=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5,
                                           factor=0.5, min_lr=1e-5, verbose=1)
    history = model.fit(
        X_tr2, y_tr2,
        validation_data=(X_val, y_val),
        epochs=100, batch_size=64,
        callbacks=[es, lr], verbose=1,
    )
    best_val = min(history.history["val_loss"])
    print(f"\n  Best val_loss: {best_val:.6f}  ({len(history.history['loss'])} epochs)")

    # 5. Save + export
    print(f"\n[5/5] Saving…")
    model.save(str(MODEL_OUT))
    print(f"  Keras model: {MODEL_OUT}")

    print(f"\n  Exporting TF.js format → {TFJS_OUT}")
    export_tfjs(model, TFJS_OUT)

    print("\n" + "=" * 60)
    print("DONE. Commit the model files and push:")
    print("  git add frontend/public/models/lstm/")
    print('  git commit -m "Add pre-trained Attention-LSTM TF.js model"')
    print("  git push")
    print("=" * 60)


if __name__ == "__main__":
    main()
