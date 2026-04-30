"""
Forecast-engine internal test harness.

Exercises each of the 6 forecast methods against three realistic synthetic
portfolio equity curves covering the space of likely user inputs:

    1. "short_low_vol"     5 years of low-vol bond-heavy drift   (~7% ann / 6% vol)
    2. "long_bull"        15 years of equity-heavy bull trend    (~12% ann / 18% vol)
    3. "short_high_vol"    3 years of concentrated-single-name   (~18% ann / 32% vol)

For each (portfolio, method) pair the harness:
    * measures wall-clock time against the production per-method budget
    * validates the ForecastBand shape (dates + 5 percentile series present)
    * validates monotonicity of the percentile band (p5 ≤ p25 ≤ p50 ≤ p75 ≤ p95)
    * prints a pass/fail summary and method-level diagnostics

Run:
    cd backend && python scripts/test_forecast_methods.py
"""

from __future__ import annotations

import sys
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from app.services.forecast_engine import (
    forecast_hmm,
    forecast_factor,
    forecast_gp,
    prepare_xgboost_features,
    prepare_nbeats_features,
    prepare_lstm_features,
)

# Per-method wall-clock budgets — mirror production timeouts.
BUDGETS_S = {
    "xgboost_features": 2.0,
    "nbeats_features":  1.0,
    "lstm_features":    1.0,
    "hmm":             45.0,
    "factor":           5.0,
    "gp":              35.0,
}


def make_equity_curve(years: float, mu_ann: float, vol_ann: float, seed: int) -> pd.Series:
    """Synthesize an equity curve via GBM with given annualised drift + vol."""
    rng = np.random.default_rng(seed)
    n   = int(years * 252)
    dt  = 1 / 252
    drift = (mu_ann - 0.5 * vol_ann ** 2) * dt
    diff  = vol_ann * math.sqrt(dt)
    rets  = drift + diff * rng.standard_normal(n)
    values = 10_000 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range(start=pd.Timestamp("2015-01-02"), periods=len(values))
    return pd.Series(values, index=idx)


def series_to_returns(s: pd.Series) -> pd.Series:
    return s.pct_change().dropna()


def validate_band(band: dict) -> tuple[bool, str]:
    required = {"dates", "p5", "p25", "p50", "p75", "p95"}
    missing  = required - set(band.keys())
    if missing:
        return False, f"missing keys: {missing}"
    n = len(band["dates"])
    for k in ("p5", "p25", "p50", "p75", "p95"):
        if len(band[k]) != n:
            return False, f"{k} length {len(band[k])} != dates length {n}"
    # Spot-check percentile ordering at horizon end
    t = -1
    a = [band["p5"][t], band["p25"][t], band["p50"][t], band["p75"][t], band["p95"][t]]
    if not all(a[i] <= a[i + 1] + 1e-6 for i in range(4)):
        return False, f"percentile ordering violated at t={t}: {a}"
    return True, "ok"


def run_case(case_name: str, equity: pd.Series) -> None:
    print(f"\n-- Case: {case_name}  (n={len(equity)} days, final=${equity.iloc[-1]:,.0f})")
    returns   = series_to_returns(equity)
    last_date = equity.index[-1].strftime("%Y-%m-%d")
    horizon   = 63
    n_paths   = 1000

    def timed(name: str, fn):
        t0 = time.time()
        try:
            out = fn()
            dt = time.time() - t0
            budget = BUDGETS_S[name]
            status = "OK " if dt <= budget else "SLOW"
            return status, dt, out, None
        except Exception as exc:
            return "FAIL", time.time() - t0, None, str(exc)

    # Feature-prep methods
    for name, fn in [
        ("xgboost_features", lambda: prepare_xgboost_features(returns, horizon, last_date)),
        ("nbeats_features",  lambda: prepare_nbeats_features(returns, horizon, last_date)),
        ("lstm_features",    lambda: prepare_lstm_features(returns, horizon, last_date)),
    ]:
        status, dt, out, err = timed(name, fn)
        if err:
            print(f"  [{status}] {name:18s} — {err}")
        else:
            extra = ""
            if name == "xgboost_features":
                extra = f" features={len(out['features'])}"
            elif name == "nbeats_features":
                extra = f" window={len(out['last_window'])}"
            elif name == "lstm_features":
                extra = f" window=({len(out['last_window'])}×{len(out['last_window'][0])})"
            print(f"  [{status}] {name:18s} {dt:6.2f}s  budget={BUDGETS_S[name]}s{extra}")

    # Band-producing methods
    for name, fn in [
        ("hmm",    lambda: forecast_hmm(returns, horizon, n_paths, last_date, None)),
        ("factor", lambda: forecast_factor(returns, horizon, n_paths, last_date, None, None)),
        ("gp",     lambda: forecast_gp(returns, horizon, last_date)),
    ]:
        status, dt, out, err = timed(name, fn)
        if err:
            print(f"  [{status}] {name:18s} {dt:6.2f}s  — {err}")
            continue
        ok, msg = validate_band(out["band"])
        flag = "PASS" if ok else "FAIL"
        p50_last = out["band"]["p50"][-1]
        print(
            f"  [{status}] {name:18s} {dt:6.2f}s  budget={BUDGETS_S[name]}s  "
            f"band={flag} ({msg})  p50({horizon}d)={p50_last:+.1f}%"
        )


def main():
    cases = [
        ("short_low_vol",   make_equity_curve(5,  0.07, 0.06, seed=11)),
        ("long_bull",       make_equity_curve(15, 0.12, 0.18, seed=22)),
        ("short_high_vol",  make_equity_curve(3,  0.18, 0.32, seed=33)),
    ]
    for name, equity in cases:
        run_case(name, equity)


if __name__ == "__main__":
    main()
