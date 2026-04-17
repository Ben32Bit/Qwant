"""
train_lstm.py — Local training script for the browser-side Attention-LSTM.

Run ONCE locally to train and export the model:
  cd backend
  pip install tensorflow tensorflowjs scikit-learn pandas yfinance
  python scripts/train_lstm.py

Outputs:
  backend/scripts/lstm_model.h5              (Keras saved model)
  frontend/public/models/lstm/model.json     (TF.js model config)
  frontend/public/models/lstm/*.bin          (TF.js weight shards)

Architecture: Attention-LSTM(64) → Dense(32) → Dense(1)  (~300-600KB)

The model is trained on a diverse 15-asset universe so it generalises
to any portfolio's feature sequences after per-portfolio MinMaxScaler
normalisation.

References
----------
CS230 Stanford (2020). Predicting Stock Market Returns Using Temporal
  Attention-Enhanced LSTM. https://cs230.stanford.edu/projects_winter_2020/reports/32066186.pdf
Bahdanau, D., Cho, K., & Bengio, Y. (2015). ICLR 2015. arXiv:1409.0473
Gal, Y. & Ghahramani, Z. (2016). Dropout as Bayesian Approximation. ICML 33.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
BACKEND_DIR  = SCRIPT_DIR.parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
TFJS_OUT     = FRONTEND_DIR / "public" / "models" / "lstm"
H5_OUT       = SCRIPT_DIR / "lstm_model.h5"

TFJS_OUT.mkdir(parents=True, exist_ok=True)

# ── Training universe ─────────────────────────────────────────────────────────
# 15 diverse assets: broad equity, bonds, commodities, international, sectors
UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",          # equity breadth
    "TLT", "AGG", "IEF",                          # fixed income
    "GLD", "USO",                                  # commodities
    "XLK", "XLF", "XLE", "XLV", "XLY",           # US sectors
]

START = "2010-01-01"
END   = "2024-12-31"

LOOKBACK    = 60     # sequence length (trading days)
TRADING_DAYS = 252
FEAT_COLS   = ["r", "vol_21d", "mom_5d", "mom_21d", "rsi_14"]


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(returns: pd.Series) -> pd.DataFrame:
    r = returns.copy()
    df = pd.DataFrame({"r": r})
    df["vol_21d"]  = r.rolling(21).std() * np.sqrt(TRADING_DAYS)
    df["mom_5d"]   = r.rolling(5).sum()
    df["mom_21d"]  = r.rolling(21).sum()
    delta          = r.diff()
    gain           = delta.clip(lower=0).rolling(14).mean()
    loss           = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"]   = gain / (gain + loss + 1e-9)
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

    Uses tf.keras.layers.Lambda(tf.reduce_sum) for the context vector sum.
    This TF op exports cleanly with tensorflowjs_converter.

    If TF.js conversion fails (rare), replace Lambda with GlobalAveragePooling1D
    on the Multiply output (approximate but TF.js-safe).
    """
    import tensorflow as tf
    from tensorflow import keras

    inputs = keras.Input(shape=(lookback, n_features), name="input")

    hidden = keras.layers.LSTM(
        64, return_sequences=True,
        dropout=0.20, recurrent_dropout=0.10,
        name="lstm_enc",
    )(inputs)  # (B, T, 64)

    # Bahdanau attention: eₜ = vᵀ · tanh(Wₐ · hₜ)
    score = keras.layers.Dense(attn_units, use_bias=False, name="attn_W")(hidden)
    score = keras.layers.Activation("tanh", name="attn_tanh")(score)
    score = keras.layers.Dense(1, use_bias=False, name="attn_v")(score)   # (B, T, 1)
    alpha = keras.layers.Softmax(axis=1, name="attn_softmax")(score)       # (B, T, 1)

    # Context vector: weighted sum over T
    # Multiply: (B, T, 64) * (B, T, 1) broadcast → (B, T, 64)
    weighted = keras.layers.Multiply(name="weighted")([hidden, alpha])
    # Sum over T axis → (B, 64)
    context = keras.layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=1), name="context_sum"
    )(weighted)

    x = keras.layers.Dropout(0.20, name="head_drop")(context)
    x = keras.layers.Dense(32, activation="relu", name="head_dense")(x)
    out = keras.layers.Dense(1, name="output")(x)

    return keras.Model(inputs=inputs, outputs=out, name="attention_lstm")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Attention-LSTM Training Script")
    print("=" * 60)

    # 1. Download price data
    print(f"\n[1/5] Downloading price data for {len(UNIVERSE)} assets ({START}→{END})…")
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("ERROR: yfinance not installed. Run: pip install yfinance")

    prices = yf.download(UNIVERSE, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    available = list(prices.columns)
    print(f"  Available: {available} ({len(prices)} trading days)")

    # 2. Build feature sequences for all assets
    print("\n[2/5] Engineering features + building sequence dataset…")
    from sklearn.preprocessing import MinMaxScaler

    all_X, all_y = [], []
    for ticker in available:
        ret = prices[ticker].pct_change().dropna()
        df  = build_features(ret)
        if len(df) < LOOKBACK + 90:
            continue

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(df[FEAT_COLS].values)

        # Chronological 70/15/15 split — no random k-fold
        n       = len(scaled)
        n_train = int(n * 0.70)
        X_tr, y_tr = make_sequences(scaled[:n_train], LOOKBACK)
        all_X.append(X_tr)
        all_y.append(y_tr)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"  Total sequences: {len(X)} (features: {X.shape[2]})")

    # 3. Build model
    print("\n[3/5] Building Attention-LSTM architecture…")
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        sys.exit("ERROR: tensorflow not installed. Run: pip install tensorflow tensorflowjs")

    tf.random.set_seed(42)
    np.random.seed(42)

    model = build_attention_lstm(LOOKBACK, len(FEAT_COLS))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.summary()
    print(f"\n  Parameters: {model.count_params():,}")

    # 4. Train
    print("\n[4/5] Training (chronological split, early stopping)…")
    # Use last 15% of the combined dataset as validation
    val_split = 0.85
    n_val = int(len(X) * (1 - val_split))
    X_tr, X_val = X[:-n_val], X[-n_val:]
    y_tr, y_val = y[:-n_val], y[-n_val:]

    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5, verbose=1)

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[es, lr],
        verbose=1,
    )

    val_loss = min(history.history["val_loss"])
    print(f"\n  Best val_loss: {val_loss:.6f}  (epochs: {len(history.history['loss'])})")

    # 5. Save Keras + export TF.js
    print(f"\n[5/5] Saving model…")
    model.save(str(H5_OUT))
    print(f"  Keras .h5: {H5_OUT}")

    print(f"\n  Exporting to TF.js format → {TFJS_OUT} …")
    try:
        import subprocess
        result = subprocess.run(
            ["tensorflowjs_converter", "--input_format=keras",
             str(H5_OUT), str(TFJS_OUT)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: tensorflowjs_converter failed:\n{result.stderr}")
            print("  Try: pip install tensorflowjs  then rerun this script.")
        else:
            tfjs_files = list(TFJS_OUT.glob("*.json")) + list(TFJS_OUT.glob("*.bin"))
            total_kb   = sum(f.stat().st_size for f in tfjs_files) // 1024
            print(f"  TF.js files: {[f.name for f in tfjs_files]}")
            print(f"  Total model size: ~{total_kb} KB")
    except FileNotFoundError:
        print("  ERROR: tensorflowjs_converter not found.")
        print("  Run: pip install tensorflowjs  then:")
        print(f"  tensorflowjs_converter --input_format=keras {H5_OUT} {TFJS_OUT}")

    print("\n" + "=" * 60)
    print("DONE. Next steps:")
    print("  1. git add frontend/public/models/lstm/")
    print('  2. git commit -m "Add pre-trained Attention-LSTM TF.js model"')
    print("  3. git push  → Vercel deploys model files as static assets")
    print("=" * 60)


if __name__ == "__main__":
    main()
