"""
inspect_timesfm.py — Probe TimesFM_2p5_200M_torch (GitHub-master install)
to discover load signature, underlying nn.Module, and LoRA target modules.

Usage on the pod:
  python inspect_timesfm.py 2>&1 | tee inspect_output.txt
"""
from __future__ import annotations

import inspect
import traceback
import torch
import timesfm

print("=" * 70)
print(f"timesfm package path: {timesfm.__file__}")
print("=" * 70)

# ── Inspect the class itself first (no load required) ───────────────────────
Cls = timesfm.TimesFM_2p5_200M_torch
print(f"\nClass: {Cls.__module__}.{Cls.__name__}")
print(f"  Base classes: {[b.__name__ for b in Cls.__mro__[1:4]]}")
print(f"  Public methods: {sorted(m for m in dir(Cls) if not m.startswith('_'))}")

# Show __init__ and from_pretrained signatures so we know how to instantiate
for method in ("__init__", "from_pretrained", "forecast", "compile"):
    fn = getattr(Cls, method, None)
    if fn:
        try:
            print(f"\n  {method}{inspect.signature(fn)}")
        except (TypeError, ValueError):
            pass

# Also print ForecastConfig signature
print(f"\nForecastConfig{inspect.signature(timesfm.ForecastConfig)}")

# ── Try to instantiate ──────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("LOAD ATTEMPTS")
print("─" * 70)

REPO = "google/timesfm-2.5-200m-pytorch"
attempts = [
    ("Cls.from_pretrained(REPO)",          lambda: Cls.from_pretrained(REPO)),
    ("Cls.from_pretrained(REPO, ...)",     lambda: Cls.from_pretrained(REPO, torch_dtype=torch.float32)),
    ("Cls()",                              lambda: Cls()),
]
m = None
for label, fn in attempts:
    try:
        m = fn()
        print(f"  [OK] {label}")
        print(f"       wrapper type: {type(m).__name__}")
        break
    except Exception as e:
        msg = str(e).split("\n")[0][:200]
        print(f"  [FAIL] {label}\n         {type(e).__name__}: {msg}")

if m is None:
    print("\nERROR: could not instantiate. Bailing.")
    raise SystemExit(1)

# ── Find underlying torch module ────────────────────────────────────────────
print("\n" + "─" * 70)
print("UNDERLYING torch.nn.Module")
print("─" * 70)
torch_model, torch_path = None, None
if isinstance(m, torch.nn.Module):
    torch_model, torch_path = m, "m  (wrapper IS the nn.Module)"
else:
    for attr in ("model", "_model", "torch_model", "_torch_model",
                 "module", "_module", "net", "decoder", "_decoder"):
        v = getattr(m, attr, None)
        if isinstance(v, torch.nn.Module):
            torch_model, torch_path = v, f"m.{attr}"
            break
    if torch_model is None:
        # exhaustive scan
        for attr in dir(m):
            if attr.startswith("_"): continue
            try:
                v = getattr(m, attr)
                if isinstance(v, torch.nn.Module):
                    torch_model, torch_path = v, f"m.{attr}"
                    break
            except Exception: pass

if torch_model is None:
    print("[FAIL] No nn.Module found on the wrapper.")
    print("All wrapper attrs:")
    for a in dir(m):
        if not a.startswith("_"):
            try: print(f"  m.{a}: {type(getattr(m, a)).__name__}")
            except Exception: pass
    raise SystemExit(1)

print(f"  Found at: {torch_path}")
print(f"  Type    : {type(torch_model).__module__}.{type(torch_model).__name__}")
print(f"  Params  : {sum(p.numel() for p in torch_model.parameters()):,}")

# ── Top-level structure ─────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("TOP-LEVEL CHILDREN")
print("─" * 70)
for name, child in torch_model.named_children():
    n = sum(p.numel() for p in child.parameters())
    print(f"  {name:30s}: {type(child).__name__:30s} ({n:>13,} params)")

# ── Find ModuleList of transformer blocks and dump one block in full ────────
print("\n" + "─" * 70)
print("ONE TRANSFORMER BLOCK (full)")
print("─" * 70)
block = None
for full_name, mod in torch_model.named_modules():
    if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 4:
        block = mod[0]
        print(f"  ModuleList at: {full_name}  (len={len(mod)})")
        print()
        break
if block is not None:
    for sub_name, sub_mod in block.named_modules():
        if not sub_name:
            print(f"  [block 0] {type(block).__name__}")
            continue
        depth = sub_name.count(".")
        indent = "    " + "  " * depth
        own = sum(p.numel() for p in sub_mod.parameters(recurse=False))
        wshape = ""
        if hasattr(sub_mod, "weight") and hasattr(sub_mod.weight, "shape"):
            wshape = f"  W={tuple(sub_mod.weight.shape)}"
        print(f"{indent}{sub_name:35s} {type(sub_mod).__name__:25s} (own:{own:>11,}){wshape}")

# ── Attention-related leaf module names (for LoRA target_modules) ───────────
print("\n" + "─" * 70)
print("LEAF MODULE NAME FREQUENCIES (last segment)")
print("─" * 70)
leaf_counts: dict[str, int] = {}
leaf_examples: dict[str, str] = {}
for full_name, mod in torch_model.named_modules():
    if any(True for _ in mod.children()):
        continue   # not a leaf
    parts = full_name.split(".")
    last = parts[-1] if parts else ""
    if not last: continue
    leaf_counts[last] = leaf_counts.get(last, 0) + 1
    leaf_examples.setdefault(last, full_name)
for leaf in sorted(leaf_counts, key=lambda k: -leaf_counts[k]):
    if leaf_counts[leaf] < 2: continue
    print(f"  {leaf:30s}  x{leaf_counts[leaf]:>3}   e.g.  {leaf_examples[leaf]}")

# ── Forward pass: try various signatures + check gradient flow ──────────────
print("\n" + "─" * 70)
print("FORWARD PASS  (gradient flow test)")
print("─" * 70)
torch_model.train()
ctx = torch.randn(2, 512, requires_grad=False)

forwards = [
    ("torch_model(ctx)",                                lambda: torch_model(ctx)),
    ("torch_model(past_values=ctx)",                    lambda: torch_model(past_values=ctx)),
    ("torch_model(input_ts=ctx)",                       lambda: torch_model(input_ts=ctx)),
    ("torch_model(ctx, horizon_len=64)",                lambda: torch_model(ctx, horizon_len=64)),
    ("torch_model(input_ts=ctx, horizon_len=64)",       lambda: torch_model(input_ts=ctx, horizon_len=64)),
    ("torch_model(ctx, prediction_length=64)",          lambda: torch_model(ctx, prediction_length=64)),
]
for label, fn in forwards:
    try:
        out = fn()
        print(f"  [OK] {label}")
        if isinstance(out, (tuple, list)):
            for i, o in enumerate(out):
                shape = getattr(o, "shape", None)
                rg = getattr(o, "requires_grad", None)
                print(f"       [{i}] type={type(o).__name__} shape={shape} requires_grad={rg}")
        elif hasattr(out, "shape"):
            print(f"       shape={out.shape} requires_grad={out.requires_grad}")
        elif hasattr(out, "keys"):
            print(f"       dict keys: {list(out.keys())}")
        else:
            print(f"       type={type(out).__name__}")
        break
    except Exception as e:
        msg = str(e).split("\n")[0][:200]
        print(f"  [FAIL] {label}\n         {type(e).__name__}: {msg}")

# ── Wrapper's inference API (for reference, not training) ───────────────────
print("\n" + "─" * 70)
print("WRAPPER INFERENCE API")
print("─" * 70)
import numpy as np
ctx_np = np.cumsum(np.random.randn(512) * 0.01).astype(np.float32)
print(f"  context shape: {ctx_np.shape}")

infer_attempts = [
    ("m.forecast(horizon=64, inputs=[ctx])",
     lambda: m.forecast(horizon=64, inputs=[ctx_np])),
    ("m.forecast([ctx], horizon=64)",
     lambda: m.forecast([ctx_np], horizon=64)),
    ("m.forecast(inputs=[ctx])",
     lambda: m.forecast(inputs=[ctx_np])),
    ("m.compile(ForecastConfig(...)) then m.forecast(...)",
     lambda: (m.compile(timesfm.ForecastConfig(max_context=512, max_horizon=64)),
              m.forecast(horizon=64, inputs=[ctx_np]))[-1]),
]
for label, fn in infer_attempts:
    try:
        out = fn()
        print(f"  [OK] {label}")
        if isinstance(out, tuple):
            for i, o in enumerate(out):
                shape = getattr(o, "shape", None) or (len(o) if hasattr(o, "__len__") else "?")
                print(f"       [{i}] type={type(o).__name__} shape/len={shape}")
        else:
            shape = getattr(out, "shape", None) or (len(out) if hasattr(out, "__len__") else "?")
            print(f"       type={type(out).__name__} shape/len={shape}")
        break
    except Exception as e:
        msg = str(e).split("\n")[0][:200]
        print(f"  [FAIL] {label}\n         {type(e).__name__}: {msg}")

print("\n" + "=" * 70)
print("END.  scp inspect_output.txt back and paste it.")
print("=" * 70)
