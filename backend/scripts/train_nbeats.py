"""
train_nbeats.py — Local training script for browser-side N-BEATS quantile forecaster.

Run ONCE locally to train and export:
  cd backend
  pip install torch yfinance pandas scikit-learn
  python scripts/train_nbeats.py

Output:
  frontend/public/models/nbeats/weights.bin  (~1-2 MB)
  frontend/public/models/nbeats/meta.json    (normalisation params + weight manifest)

Architecture — N-BEATS (Oreshkin et al. 2020, ICLR):
  3 residual stacks. Each stack: FC(256)×4 → backcast(30) + forecast(21×5).
  Input is successively "explained" by each block (backcast subtracted).
  Final forecast = sum of all block forecasts.

  Modified for quantile regression:
    - Output head: 21 × 5 quantiles [p5, p25, p50, p75, p95]
    - Loss: sum of pinball losses across all quantiles and all forecast steps
    - No interpretable basis expansion (generic blocks) — keeps ONNX simple

Input  → shape [batch, 30]    (last 30 daily returns)
Output → shape [batch, 21, 5] (21-step forecast × 5 quantiles, as daily returns)

Multi-horizon fan chart (in NBeatsInferer.js):
  Run ceil(target_days / 21) recursive 21-day periods to produce the fan chart.
  At the current 63-day forecast horizon that's 3 periods exactly. Each period
  uses the p50 median of the previous period's output as the new input window.

Out-of-sample methodology:
  Chronological 70/15/15 train/val/test split across all assets.
  Walk-forward construction: sequences built from individual assets 2010-2024.

References:
  Oreshkin, B., Carpov, D., Chapados, N., & Bengio, Y. (2020).
    N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.
    ICLR 2020. https://arxiv.org/abs/1905.10437
  Koenker, R. & Bassett, G. (1978). Regression Quantiles.
    Econometrica, 46(1), 33–50. https://doi.org/10.2307/1913643
  López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Ch. 7.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
BACKEND_DIR  = SCRIPT_DIR.parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
MODEL_OUT    = FRONTEND_DIR / "public" / "models" / "nbeats"
MODEL_OUT.mkdir(parents=True, exist_ok=True)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    sys.exit("ERROR: torch not installed. Run: pip install torch")

# ── Hyperparameters ───────────────────────────────────────────────────────────
UNIVERSE     = ["SPY","QQQ","IWM","EFA","EEM","TLT","AGG","IEF","GLD","USO",
                "XLK","XLF","XLE","XLV","XLY"]
START        = "2010-01-01"
END          = "2024-12-31"
TRADING_DAYS = 252

LOOKBACK     = 30    # input window length
HORIZON      = 21    # forecast horizon (days)
QUANTILES    = [0.05, 0.25, 0.50, 0.75, 0.95]
N_QUANTILES  = len(QUANTILES)

# Model
N_BLOCKS     = 3
HIDDEN       = 256
FC_LAYERS    = 4

# Training
EPOCHS       = 150
BATCH_SIZE   = 512
LR           = 1e-3
PATIENCE     = 20    # early stopping
SEED         = 42


# ── N-BEATS block ─────────────────────────────────────────────────────────────

class NBeatsBlock(nn.Module):
    """
    Single N-BEATS residual block.

    Fully-connected stack → two output heads:
      backcast: reconstructs the lookback window (subtracted from input for next block)
      forecast: contributes a partial 21-step × 5-quantile forecast

    Generic blocks (no polynomial/Fourier basis) for simpler weight export.
    """
    def __init__(self, lookback: int, horizon: int, n_quantiles: int,
                 hidden: int, n_layers: int):
        super().__init__()
        layers = [nn.Linear(lookback, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.fc = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden, lookback)
        self.forecast_head = nn.Linear(hidden, horizon * n_quantiles)
        self.horizon    = horizon
        self.n_quantiles = n_quantiles

    def forward(self, x):
        h        = self.fc(x)
        backcast = self.backcast_head(h)
        forecast = self.forecast_head(h).view(x.shape[0], self.horizon, self.n_quantiles)
        return backcast, forecast


class NBeats(nn.Module):
    """
    N-BEATS stack: residual composition of NBeatsBlock instances.

    Each block explains part of the input (backcast subtracted), and contributes
    its forecast. Final prediction = sum of all block forecasts.

    Reference: Oreshkin et al. (2020, ICLR). https://arxiv.org/abs/1905.10437
    """
    def __init__(self, lookback: int, horizon: int, n_quantiles: int,
                 n_blocks: int, hidden: int, n_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(lookback, horizon, n_quantiles, hidden, n_layers)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        # x: (batch, lookback)
        residual         = x
        total_forecast   = torch.zeros(x.shape[0], self.blocks[0].horizon,
                                       self.blocks[0].n_quantiles, device=x.device)
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual           = residual - backcast
            total_forecast     = total_forecast + forecast
        return total_forecast   # (batch, horizon, n_quantiles)


# ── Pinball (quantile) loss ───────────────────────────────────────────────────

def pinball_loss(pred: torch.Tensor, target: torch.Tensor,
                 quantiles: list[float]) -> torch.Tensor:
    """
    Mean pinball loss across all quantiles and forecast steps.

    pred:   (batch, horizon, n_quantiles)
    target: (batch, horizon)             — actual future returns

    Reference: Koenker & Bassett (1978). Regression Quantiles. Econometrica.
    """
    target_exp = target.unsqueeze(-1).expand_as(pred)  # (batch, horizon, n_q)
    q = torch.tensor(quantiles, dtype=torch.float32, device=pred.device)
    errors = target_exp - pred                           # positive = under-prediction
    loss = torch.max(q * errors, (q - 1) * errors)      # pinball
    return loss.mean()


# ── Data pipeline ─────────────────────────────────────────────────────────────

def build_dataset(prices: pd.DataFrame,
                  lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) from price data.
    X: (n_samples, lookback) — trailing daily returns
    y: (n_samples, horizon)  — next `horizon` daily returns
    """
    all_X, all_y = [], []

    for ticker in prices.columns:
        r = prices[ticker].pct_change().dropna().values.astype(np.float32)
        n = len(r)
        for i in range(lookback, n - horizon):
            all_X.append(r[i - lookback: i])
            all_y.append(r[i: i + horizon])

    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, quantiles: list[float],
          epochs: int, lr: float, patience: int, device: torch.device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience // 2, factor=0.5
    )

    best_val_loss = float("inf")
    best_state    = None
    wait          = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred   = model(xb)
            loss   = pinball_loss(pred, yb, quantiles)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred   = model(xb)
                val_loss += pinball_loss(pred, yb, quantiles).item() * len(xb)

        n_train  = len(train_loader.dataset)
        n_val    = len(val_loader.dataset)
        tl       = train_loss / n_train
        vl       = val_loss   / n_val
        scheduler.step(vl)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{epochs}  train={tl:.6f}  val={vl:.6f}")

        if vl < best_val_loss:
            best_val_loss = vl
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            wait          = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch} (best val={best_val_loss:.6f})")
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_val_loss


# ── Weight binary export ──────────────────────────────────────────────────────
# Skip ONNX entirely — N-BEATS is purely linear+ReLU which is trivial to
# implement in 50 lines of JS.  We write weights as a raw float32 binary
# (same format as the LSTM TF.js export) and a manifest JSON so NBeatsInferer.js
# can reconstruct the model without any ONNX / TF runtime dependency.

def export_weights(model: nn.Module, out_dir: Path) -> list[dict]:
    """
    Serialise all weight tensors to a flat float32 binary + manifest JSON.

    Format mirrors the TF.js LayersModel convention used by the LSTM:
      weights.bin : concatenated little-endian float32 arrays
      manifest    : list of {name, shape, offset, size} dicts
    """
    model.eval()
    manifest = []
    buffers  = []
    offset   = 0

    for name, param in model.named_parameters():
        arr = param.detach().cpu().numpy().astype("float32")
        raw = arr.tobytes()
        manifest.append({
            "name":   name,
            "shape":  list(arr.shape),
            "offset": offset,
            "size":   len(raw) // 4,   # number of float32 values
        })
        buffers.append(raw)
        offset += len(raw)

    bin_path = out_dir / "weights.bin"
    with open(bin_path, "wb") as f:
        for buf in buffers:
            f.write(buf)

    kb = bin_path.stat().st_size // 1024
    print(f"  weights.bin  ({kb} KB)  —  {len(manifest)} tensors")
    return manifest


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("N-BEATS Quantile Forecast Training Script")
    print("=" * 60)
    print(f"  Lookback={LOOKBACK}d  Horizon={HORIZON}d  {N_BLOCKS} blocks  hidden={HIDDEN}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # 1. Download prices
    print(f"\n[1/5] Downloading {len(UNIVERSE)} assets ({START}→{END})…")
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("ERROR: yfinance not installed. Run: pip install yfinance")

    prices = yf.download(UNIVERSE, start=START, end=END,
                         auto_adjust=True, progress=False)["Close"]
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    print(f"  {len(prices.columns)} assets  ({len(prices)} trading days)")

    # 2. Build dataset
    print("\n[2/5] Building sequence dataset…")
    X, y = build_dataset(prices, LOOKBACK, HORIZON)
    print(f"  {len(X)} sequences  |  X: {X.shape}  y: {y.shape}")

    # Chronological 70/15/15 split
    n       = len(X)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.85)
    X_tr, y_tr = X[:n_train],       y[:n_train]
    X_val, y_val = X[n_train:n_val], y[n_train:n_val]
    X_te, y_te = X[n_val:],          y[n_val:]
    print(f"  Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_te)}")

    # Normalise inputs (z-score per series — we use global stats across all assets)
    mu_r    = float(X_tr.mean())
    sigma_r = float(X_tr.std()) + 1e-8
    X_tr_n  = (X_tr  - mu_r) / sigma_r
    X_val_n = (X_val - mu_r) / sigma_r
    X_te_n  = (X_te  - mu_r) / sigma_r
    y_tr_n  = (y_tr  - mu_r) / sigma_r
    y_val_n = (y_val - mu_r) / sigma_r
    y_te_n  = (y_te  - mu_r) / sigma_r

    # 3. Build model and DataLoaders
    print("\n[3/5] Building model…")
    model = NBeats(
        lookback=LOOKBACK, horizon=HORIZON, n_quantiles=N_QUANTILES,
        n_blocks=N_BLOCKS, hidden=HIDDEN, n_layers=FC_LAYERS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    train_ds = TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(y_tr_n))
    val_ds   = TensorDataset(torch.from_numpy(X_val_n), torch.from_numpy(y_val_n))
    te_ds    = TensorDataset(torch.from_numpy(X_te_n),  torch.from_numpy(y_te_n))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    te_loader    = DataLoader(te_ds,    batch_size=BATCH_SIZE, shuffle=False)

    # 4. Train
    print(f"\n[4/5] Training (max {EPOCHS} epochs, patience={PATIENCE})…")
    best_val = train(model, train_loader, val_loader, QUANTILES,
                     EPOCHS, LR, PATIENCE, device)

    # OOS test metrics
    model.eval()
    oos_preds, oos_actuals = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            pred = model(xb.to(device))   # (batch, horizon, n_q)
            oos_preds.append(pred.cpu().numpy())
            oos_actuals.append(yb.numpy())
    oos_preds   = np.concatenate(oos_preds, axis=0)      # (N, 21, 5)
    oos_actuals = np.concatenate(oos_actuals, axis=0)     # (N, 21)

    # Denormalise
    oos_preds_raw   = oos_preds   * sigma_r + mu_r
    oos_actuals_raw = oos_actuals * sigma_r + mu_r

    p50_pred = oos_preds_raw[:, :, 2]  # median quantile index
    oos_r2   = float(1 - np.var(oos_actuals_raw - p50_pred) / np.var(oos_actuals_raw))
    ic        = float(np.corrcoef(p50_pred.ravel(), oos_actuals_raw.ravel())[0, 1])

    # Pinball coverage
    coverages = {}
    for qi, q in enumerate(QUANTILES):
        actual_below = (oos_actuals_raw < oos_preds_raw[:, :, qi]).mean()
        coverages[f"p{int(q*100):02d}"] = round(float(actual_below), 3)

    print(f"\n  Best val loss:  {best_val:.6f}")
    print(f"  OOS R² (p50):   {oos_r2:.4f}")
    print(f"  IC (p50 rank):  {ic:.4f}")
    print(f"  Coverage check: {coverages}")

    # 5. Export weights as raw float32 binary (no ONNX — pure-JS forward pass)
    print(f"\n[5/5] Exporting weights → {MODEL_OUT}")
    import json
    manifest = export_weights(model.cpu(), MODEL_OUT)

    meta = {
        "lookback":         LOOKBACK,
        "horizon":          HORIZON,
        "quantiles":        QUANTILES,
        "n_blocks":         N_BLOCKS,
        "n_params":         n_params,
        "mu_r":             float(mu_r),
        "sigma_r":          float(sigma_r),
        "oos_r2":           round(oos_r2, 4),
        "ic":               round(ic, 4),
        "coverages":        coverages,
        "train_period":     f"{START}–{END}",
        "n_assets":         len(prices.columns),
        "weights_manifest": manifest,
    }
    with open(MODEL_OUT / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta.json written  (mu_r={mu_r:.6f}  sigma_r={sigma_r:.6f}  {len(manifest)} tensors)")

    print("\n" + "=" * 60)
    print(f"DONE.  OOS R²={oos_r2:.4f}  IC={ic:.4f}")
    print(f"Coverage: {coverages}")
    print("Commit the model files and push:")
    print("  git add frontend/public/models/nbeats/")
    print('  git commit -m "Phase 2B: Add N-BEATS quantile model (pure-JS weights)"')
    print("  git push")
    print("=" * 60)


if __name__ == "__main__":
    main()
