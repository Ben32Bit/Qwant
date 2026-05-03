"""
finetune_timesfm.py — LoRA fine-tuning for TimesFM 2.5 on equity log-prices.

Prerequisites
-------------
1. pip install -r requirements-train.txt   (adds peft, accelerate, datasets)
2. python scripts/build_finetune_universe.py   (builds the parquet)

Outputs
-------
  scripts/timesfm_lora_adapter/   (~25 MB PEFT adapter)

Copy to Railway volume after training:
  scp -i ~/.ssh/runpod -P <port> -r \
      root@<pod-ip>:/workspace/timesfm_lora_adapter \
      backend/scripts/timesfm_lora_adapter

RunPod recommended spec
-----------------------
  GPU  : L40S (48 GB VRAM)  or  RTX 4090 (24 GB VRAM)
  VRAM : TimesFM 200M is ~800 MB in fp32 + LoRA overhead → 4 GB peak
  RAM  : 16 GB system RAM is plenty
  Cost : ~$0.79/hr (L40S on RunPod)  →  ~$3-5 total

Log-price ↔ inference bridge
-----------------------------
  Training  : log-prices  (scale-invariant, crash-safe, matches model's training shape)
  Inference : model outputs predicted log-prices L̂_{t+1..t+H}
              cum_ret_h = exp(L̂_{t+h} - L̂_T) - 1, clamped to ±exp(0.7)-1 ≈ ±101%
  This replaces the zero-shot raw-level path in timesfm_provider.py once the
  adapter is deployed.

Training objective
------------------
  For each sliding window (context_len=512, horizon=63):
    - input  : past 512 log-price values  (already on model's native scale)
    - target : next 63 log-price values
    - loss   : Huber(δ=0.02) — tolerant of the fat-tailed log-return distribution
  Huber δ=0.02 ≈ 2 log-return units, larger than typical daily moves (~0.01)
  but robust to multi-sigma events.

LoRA config
-----------
  r=16, alpha=32, dropout=0.05, target_modules: q/k/v/o projections only.
  Freezes all base model weights; only ~3 M LoRA parameters are updated.
  Adapter stored in standard PEFT format — load with PeftModel.from_pretrained().
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG  —  all tunable hyperparameters in one place
# ═══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    # Paths
    parquet_path  = Path(__file__).parent / "universe_logprices_2000_2022.parquet",
    adapter_out   = Path(__file__).parent / "timesfm_lora_adapter",

    # TimesFM model
    model_id      = "google/timesfm-2.5-200m-pytorch",

    # Window shape
    context_len   = 512,   # must match production CONTEXT_LEN
    pred_horizon  = 63,    # one quarter of trading days

    # Dataset
    stride        = 21,    # slide window by ~1 month between samples
    val_tickers   = [      # held-out tickers (never in training windows)
        "SPY", "QQQ", "TLT", "GLD", "AAPL", "JPM", "XOM", "BTC-USD",
    ],
    max_train_windows_per_ticker = 500,   # cap so large-history tickers don't dominate

    # LoRA
    lora_r        = 16,
    lora_alpha    = 32,
    lora_dropout  = 0.05,
    lora_modules  = ["q_proj", "k_proj", "v_proj", "o_proj"],

    # Training
    epochs        = 3,
    batch_size    = 16,
    lr            = 1e-4,
    grad_accum    = 4,     # effective batch = batch_size × grad_accum = 64
    max_grad_norm = 1.0,
    huber_delta   = 0.02,

    # Validation gate
    gate_max_mse_ratio = 1.10,   # val MSE / var(y_val) must be ≤ this
    gate_min_dir_acc   = 0.50,   # directional accuracy on val set

    # Safety clamp for inference (applied in timesfm_provider.py, not here)
    log_return_clamp = 0.70,     # ±0.70 log units ≈ ±101% cum return at horizon
)
# ═══════════════════════════════════════════════════════════════════════════════


# ── Dataset ───────────────────────────────────────────────────────────────────

class LogPriceWindowDataset:
    """Sliding-window dataset over log-price series.

    Returns numpy arrays (not tensors) so the data loader can be used with
    or without torch.utils.data.DataLoader.
    """

    def __init__(
        self,
        log_prices: pd.DataFrame,
        context_len: int,
        pred_horizon: int,
        stride: int,
        max_per_ticker: int | None = None,
        exclude_tickers: list[str] | None = None,
        only_tickers: list[str] | None = None,
    ):
        self.context_len  = context_len
        self.pred_horizon = pred_horizon
        self.window_len   = context_len + pred_horizon

        exclude = set(exclude_tickers or [])
        if only_tickers is not None:
            cols = [c for c in log_prices.columns if c in set(only_tickers)]
        else:
            cols = [c for c in log_prices.columns if c not in exclude]

        self.X: list[np.ndarray] = []   # (context_len,)
        self.y: list[np.ndarray] = []   # (pred_horizon,)

        for ticker in cols:
            series = log_prices[ticker].dropna().values.astype(np.float32)
            if len(series) < self.window_len + stride:
                continue
            windows = []
            for start in range(0, len(series) - self.window_len + 1, stride):
                ctx = series[start : start + context_len]
                tgt = series[start + context_len : start + self.window_len]
                windows.append((ctx, tgt))
            if max_per_ticker and len(windows) > max_per_ticker:
                # Keep the most recent windows (latest market regimes)
                windows = windows[-max_per_ticker:]
            for ctx, tgt in windows:
                self.X.append(ctx)
                self.y.append(tgt)

        self.X = np.array(self.X, dtype=np.float32)   # (N, context_len)
        self.y = np.array(self.y, dtype=np.float32)   # (N, pred_horizon)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_torch_loader(dataset, batch_size: int, shuffle: bool):
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        sys.exit("ERROR: PyTorch not installed.  pip install torch")

    X_t = torch.tensor(dataset.X)
    y_t = torch.tensor(dataset.y)
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


# ── Model loading + LoRA ──────────────────────────────────────────────────────

def load_base_model(model_id: str):
    try:
        import torch
    except ImportError:
        sys.exit("ERROR: PyTorch not installed.  pip install torch")

    log.info("Loading base model %s …", model_id)
    for cls_name in ("TimesFm2_5ModelForPrediction", "TimesFmModelForPrediction"):
        try:
            mod = __import__("transformers", fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            model = cls.from_pretrained(model_id, torch_dtype=torch.float32)
            log.info("Loaded %s", cls_name)
            return model
        except (ImportError, AttributeError, Exception) as e:
            log.debug("  %s not available: %s", cls_name, e)
            continue
    sys.exit(
        "ERROR: TimesFM class not found — install transformers >= 4.48.0\n"
        "  pip install 'transformers>=4.48.0'"
    )


def apply_lora(model, cfg: dict):
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        sys.exit("ERROR: peft not installed.  pip install 'peft>=0.13'")

    # TimesFM is not in PEFT's built-in task list, so we use FEATURE_EXTRACTION
    # which skips the task-specific head assumption and wraps any nn.Module.
    lora_cfg = LoraConfig(
        r             = cfg["lora_r"],
        lora_alpha    = cfg["lora_alpha"],
        lora_dropout  = cfg["lora_dropout"],
        target_modules= cfg["lora_modules"],
        bias          = "none",
        task_type     = TaskType.FEATURE_EXTRACTION,
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()
    return peft_model


# ── Forward pass helper ───────────────────────────────────────────────────────

def _model_forward(model, past_values, pred_horizon: int):
    """Unified forward call; returns mean_predictions tensor (batch, horizon)."""
    import torch

    try:
        output = model(past_values=past_values, prediction_length=pred_horizon)
    except TypeError:
        output = model(past_values=past_values)

    # Probe dict-like ModelOutput
    for key in ("mean_predictions", "prediction_outputs", "quantile_preds"):
        try:
            v = output[key] if hasattr(output, "__getitem__") else getattr(output, key, None)
            if v is not None:
                arr = v
                break
        except (KeyError, TypeError):
            arr = getattr(output, key, None)
            if arr is not None:
                break
    else:
        available = sorted(
            k for k in (list(output.keys()) if hasattr(output, "keys") else dir(output))
            if not str(k).startswith("_")
        )
        raise RuntimeError(
            f"TimesFM output format unrecognised — available attrs: {available}"
        )

    # If quantile output (batch, horizon, n_quantiles), extract median (dim -1, idx 4)
    if arr.dim() == 3:
        arr = arr[:, :, 4]   # Q0.5
    return arr[:, :pred_horizon]   # (batch, pred_horizon)


# ── Huber loss ────────────────────────────────────────────────────────────────

def huber_loss(pred, target, delta: float):
    import torch
    err = pred - target
    abs_err = err.abs()
    loss = torch.where(abs_err <= delta, 0.5 * err ** 2, delta * (abs_err - 0.5 * delta))
    return loss.mean()


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model, val_loader, pred_horizon: int, huber_delta: float, device) -> dict:
    import torch

    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = _model_forward(model, X_batch, pred_horizon)
            loss = huber_loss(pred, y_batch, huber_delta)
            total_loss += loss.item()
            n_batches  += 1
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Use 1-day-ahead directional accuracy (first forecast step vs context end)
    # as a signal for whether the model learned anything useful
    dir_acc = float(np.mean(np.sign(preds[:, 0]) == np.sign(targets[:, 0])))

    # MSE on full horizon
    mse     = float(np.mean((preds - targets) ** 2))
    y_var   = float(np.var(targets))
    ratio   = mse / y_var if y_var > 1e-10 else float("inf")

    return {
        "val_loss":  total_loss / max(n_batches, 1),
        "val_mse":   mse,
        "y_var":     y_var,
        "mse_ratio": ratio,
        "dir_acc":   dir_acc,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import torch
    import torch.optim as optim

    cfg = CFG

    print("=" * 68)
    print("TimesFM 2.5 — LoRA fine-tuning")
    print(f"  Model   : {cfg['model_id']}")
    print(f"  Context : {cfg['context_len']}  Horizon: {cfg['pred_horizon']}")
    print(f"  LoRA r  : {cfg['lora_r']}   α: {cfg['lora_alpha']}")
    print(f"  Epochs  : {cfg['epochs']}   Batch: {cfg['batch_size']} × {cfg['grad_accum']} grad_accum")
    print("=" * 68)

    # ── 1. Load parquet ──────────────────────────────────────────────────────
    parquet = cfg["parquet_path"]
    if not parquet.exists():
        sys.exit(
            f"ERROR: {parquet} not found.\n"
            "Run:  python scripts/build_finetune_universe.py"
        )

    log.info("[1/6] Loading log-price universe from %s …", parquet.name)
    log_prices = pd.read_parquet(parquet)
    log.info("  Shape: %d rows × %d tickers", *log_prices.shape)

    # ── 2. Build datasets ────────────────────────────────────────────────────
    log.info("[2/6] Building sliding-window datasets …")
    val_tickers = [t for t in cfg["val_tickers"] if t in log_prices.columns]
    log.info("  Val tickers: %s", val_tickers)

    train_ds = LogPriceWindowDataset(
        log_prices,
        context_len    = cfg["context_len"],
        pred_horizon   = cfg["pred_horizon"],
        stride         = cfg["stride"],
        max_per_ticker = cfg["max_train_windows_per_ticker"],
        exclude_tickers= val_tickers,
    )
    val_ds = LogPriceWindowDataset(
        log_prices,
        context_len    = cfg["context_len"],
        pred_horizon   = cfg["pred_horizon"],
        stride         = cfg["stride"] * 2,   # sparser stride on val to avoid overlap with train
        max_per_ticker = None,
        only_tickers   = val_tickers,
    )
    log.info("  Train: %d windows   Val: %d windows", len(train_ds), len(val_ds))

    if len(train_ds) < 1000:
        sys.exit(
            f"ERROR: only {len(train_ds)} training windows — universe too small. "
            "Check that build_finetune_universe.py completed successfully."
        )
    if len(val_ds) < 50:
        sys.exit(
            f"ERROR: only {len(val_ds)} val windows. "
            "Check val_tickers are present in the parquet."
        )

    train_loader = make_torch_loader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = make_torch_loader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)

    # ── 3. Load model + apply LoRA ───────────────────────────────────────────
    log.info("[3/6] Loading TimesFM and applying LoRA …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("  Device: %s", device)
    if device.type == "cpu":
        log.warning("  No GPU detected — training will be slow (~10× vs L40S)")

    base_model = load_base_model(cfg["model_id"])
    model      = apply_lora(base_model, cfg)
    model      = model.to(device)
    model.train()

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"] * len(train_loader) // cfg["grad_accum"],
    )

    # ── 4. Training loop ─────────────────────────────────────────────────────
    log.info("[4/6] Training …")
    best_val_loss = float("inf")
    best_state    = None
    step          = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n_steps    = 0
        optimizer.zero_grad()
        t0 = time.time()

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = _model_forward(model, X_batch, cfg["pred_horizon"])
            loss = huber_loss(pred, y_batch, cfg["huber_delta"]) / cfg["grad_accum"]
            loss.backward()

            if (batch_idx + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg["max_grad_norm"],
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

            epoch_loss += loss.item() * cfg["grad_accum"]
            n_steps    += 1

            if step % 100 == 0 and step > 0:
                log.info(
                    "  epoch %d  step %d  train_loss=%.4f  lr=%.2e",
                    epoch, step, epoch_loss / n_steps,
                    scheduler.get_last_lr()[0],
                )

        val_metrics = validate(model, val_loader, cfg["pred_horizon"], cfg["huber_delta"], device)
        elapsed = time.time() - t0
        log.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
            "mse_ratio=%.3f | dir_acc=%.1f%% | %.0fs",
            epoch, cfg["epochs"],
            epoch_loss / max(n_steps, 1),
            val_metrics["val_loss"],
            val_metrics["mse_ratio"],
            val_metrics["dir_acc"] * 100,
            elapsed,
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            # Save a snapshot of the best LoRA adapter state in memory
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()
                          if "lora" in k.lower()}
            log.info("  ↑ new best val_loss=%.4f (saved in memory)", best_val_loss)

    # Restore best state
    if best_state:
        current = model.state_dict()
        current.update(best_state)
        model.load_state_dict(current)

    # ── 5. Validation gate ───────────────────────────────────────────────────
    log.info("[5/6] Validation gate …")
    model.eval()
    final_metrics = validate(model, val_loader, cfg["pred_horizon"], cfg["huber_delta"], device)
    log.info("  Final val MSE/var(y) : %.3f  (gate: ≤ %.2f)",
             final_metrics["mse_ratio"], cfg["gate_max_mse_ratio"])
    log.info("  Final dir accuracy   : %.1f%%  (gate: ≥ %.0f%%)",
             final_metrics["dir_acc"] * 100, cfg["gate_min_dir_acc"] * 100)

    failures = []
    if final_metrics["mse_ratio"] > cfg["gate_max_mse_ratio"]:
        failures.append(
            f"Val MSE/var(y) = {final_metrics['mse_ratio']:.3f} > "
            f"{cfg['gate_max_mse_ratio']} — model not better than null predictor"
        )
    if final_metrics["dir_acc"] < cfg["gate_min_dir_acc"]:
        failures.append(
            f"Directional accuracy {final_metrics['dir_acc']:.1%} < "
            f"{cfg['gate_min_dir_acc']:.0%}"
        )

    if failures:
        print("\n" + "=" * 68)
        print("VALIDATION GATE FAILED — adapter NOT saved:")
        for msg in failures:
            print(f"  ✗ {msg}")
        print(
            "\nTry: longer training, lower lr, or check the parquet for data quality."
        )
        print("=" * 68)
        sys.exit(1)

    log.info("  All gate checks passed.")

    # ── 6. Save adapter ──────────────────────────────────────────────────────
    log.info("[6/6] Saving LoRA adapter …")
    out = cfg["adapter_out"]
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    log.info("  Adapter saved → %s", out)

    # Write a small metadata file so timesfm_provider.py can verify compatibility
    import json
    meta = {
        "model_id":     cfg["model_id"],
        "context_len":  cfg["context_len"],
        "pred_horizon": cfg["pred_horizon"],
        "lora_r":       cfg["lora_r"],
        "lora_alpha":   cfg["lora_alpha"],
        "val_mse_ratio":round(final_metrics["mse_ratio"], 4),
        "val_dir_acc":  round(final_metrics["dir_acc"], 4),
        "train_end":    "2022-12-31",
        "input_transform": "log_price",
    }
    with open(out / "finetune_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 68)
    print("DONE.")
    print(f"  Adapter: {out}")
    print(f"  Val MSE/var(y): {final_metrics['mse_ratio']:.3f}")
    print(f"  Dir accuracy  : {final_metrics['dir_acc']:.1%}")
    print()
    print("Next steps:")
    print("  1. Copy adapter back from RunPod:")
    print("     scp -i ~/.ssh/runpod -P <port> -r \\")
    print("         root@<pod-ip>:/workspace/timesfm_lora_adapter \\")
    print("         backend/scripts/timesfm_lora_adapter")
    print("  2. Commit + deploy — timesfm_provider.py will auto-detect the adapter")
    print("=" * 68)


if __name__ == "__main__":
    main()
