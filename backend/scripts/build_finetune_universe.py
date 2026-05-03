"""
build_finetune_universe.py — Download the TimesFM fine-tuning universe.

Pulls ~1,100 US equity + ETF tickers (S&P 500 + supplemental mid-caps + ETFs)
from 2000-01-01 → 2022-12-31, converts to log-prices, and writes:

    backend/scripts/universe_logprices_2000_2022.parquet

2023-2025 is held out for zero-shot OOS benchmarking.

Usage:
  python scripts/build_finetune_universe.py [--force]

  --force   Re-download even if a fresh parquet already exists.

Run time: 20-40 min on first call (Yahoo rate limits ~1 req/0.3s).
Subsequent calls within 7 days are a no-op (cache check).

Log-prices rationale
--------------------
We store log(price) rather than raw levels for training because:
  * Scale-invariant across assets (AAPL at 150, SPY at 450, TLT at 90 all
    become numbers of similar magnitude in log space).
  * TimesFM is trained on diverse level series — log-prices match that shape
    while eliminating cross-asset scale differences that would bias learning.
  * At inference time the fine-tuned model outputs future log-prices; we
    convert via exp(pred_log - last_log) - 1 with a ±0.7 log-unit safety
    clamp to get bounded cumulative returns.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("ERROR: yfinance not installed.  pip install yfinance>=0.2.50")

SCRIPT_DIR = Path(__file__).parent

# ── Date range ────────────────────────────────────────────────────────────────
START = "2000-01-01"
END   = "2022-12-31"   # 2023-2025 held out for OOS evaluation

# ── Output ────────────────────────────────────────────────────────────────────
OUT_PATH    = SCRIPT_DIR / "universe_logprices_2000_2022.parquet"
CACHE_DAYS  = 7          # re-use existing parquet if < 7 days old
MIN_ROWS    = 252        # drop tickers with < 1 year of data
MIN_TICKERS = 200        # abort if fewer than this survive

# ── Universe ──────────────────────────────────────────────────────────────────
# S&P 500 is fetched dynamically from Wikipedia (reliable, stable URL).
# SUPPLEMENTAL covers Russell 1000 mid-caps not typically in S&P 500 + all
# major ETF categories so the model sees bond/commodity/international dynamics.
SUPPLEMENTAL: list[str] = [
    # ── Broad equity ETFs ──────────────────────────────────────────────────
    "SPY", "QQQ", "IWM", "VTI", "VT", "EFA", "EEM", "VEA", "VWO",
    "ACWI", "IEFA", "IEMG", "SCHB", "ITOT", "SCHA", "SCHX", "SCHD",
    # ── Sector ETFs ───────────────────────────────────────────────────────
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "XLI", "XLB", "XLRE",
    "XBI", "IBB", "GDX", "GDXJ", "ITB", "XHB", "XRT", "XME", "KBE", "KRE",
    "SOXX", "SMH", "VGT", "VHT", "VFH", "VDE", "VDC", "VPU",
    # ── Fixed income ETFs ─────────────────────────────────────────────────
    "TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "JNK", "TIP",
    "BNDX", "SHV", "GOVT", "MUB", "BIV", "BSV", "VCSH", "VCIT",
    "EMB", "FALN", "FLOT", "JPST",
    # ── Commodity ETFs ────────────────────────────────────────────────────
    "GLD", "SLV", "IAU", "USO", "UNG", "DBC", "PDBC", "CORN", "WEAT", "SOYB",
    # ── Leveraged / inverse ETFs ──────────────────────────────────────────
    "TQQQ", "UPRO", "SOXL", "TECL", "LABU", "UDOW",
    "SQQQ", "SPXS", "SDOW", "VIXY",
    # ── VIX products ──────────────────────────────────────────────────────
    "VXX", "UVXY",
    # ── International ETFs ────────────────────────────────────────────────
    "EWJ", "EWZ", "EWG", "EWU", "EWC", "EWH", "EWA", "EWS", "EWT",
    "FXI", "MCHI", "INDA", "EWY", "EZU",
    # ── Real estate ETFs ──────────────────────────────────────────────────
    "VNQ", "IYR", "SCHH", "REM",
    # ── Crypto proxy ──────────────────────────────────────────────────────
    "BTC-USD", "ETH-USD",
    # ── Large-cap non-US ADRs (long history, liquid) ──────────────────────
    "TSM", "ASML", "SAP", "NVO", "AZN", "RIO", "BP", "HSBC", "TD", "BHP",
    "TM", "SNY", "BABA",
    # ── Notable mid-caps / growth stocks (not always in S&P 500) ─────────
    "IBKR", "WING", "CAVA", "TXRH", "DECK", "WST", "AAON", "EXPO", "SITE",
    "LGND", "MMSI", "HALO", "MEDP", "CVLT", "ALRM", "QLYS", "CVCO",
    "LNTH", "FRPT", "TOST",
    # ── Cloud / SaaS (post-2015, shorter history — good for generalization)
    "DDOG", "NET", "SNOW", "CRWD", "MDB", "CFLT", "GTLB", "ZS", "OKTA",
    "PANW", "SMAR", "BILL", "ZM", "TWLO", "DOCU",
    # ── Consumer / retail ─────────────────────────────────────────────────
    "ETSY", "HOOD", "SOFI", "AFRM", "OPEN",
    # ── Large-cap breadth (many already in S&P 500 — dedup is fine) ──────
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "SCHW",
    "BRK-B", "V", "MA", "PYPL", "AXP",
    "UNH", "CVS", "CI", "HUM", "MOH", "CNC",
    "JNJ", "PFE", "MRK", "BMY", "ABBV", "AMGN", "GILD", "BIIB", "REGN", "VRTX",
    "XOM", "CVX", "SLB", "HAL", "COP", "EOG",
    "HD", "LOW", "TGT", "WMT", "COST",
    "MCD", "SBUX", "YUM", "CMG", "DRI",
    "NKE", "DIS", "NFLX", "CMCSA",
    "T", "VZ", "TMUS",
    "NEE", "DUK", "SO", "AEP", "EXC",
    "AMT", "PLD", "CCI", "EQIX", "DLR",
    "CAT", "DE", "HON", "MMM", "GE", "BA", "RTX", "LMT", "NOC", "GD",
    "LIN", "APD", "ECL", "DOW", "NUE", "FCX",
    "ORCL", "INTU", "ADP", "PAYX",
    "AVGO", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "INTC", "AMD", "MU",
    "CRM", "NOW", "WDAY", "TEAM",
    "UBER", "LYFT", "ABNB", "BKNG", "EXPE",
]


def _fetch_sp500_tickers() -> list[str]:
    """Pull the current S&P 500 constituent list from Wikipedia."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"  S&P 500: {len(tickers)} tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"  WARNING: could not fetch S&P 500 from Wikipedia ({e}) — skipping")
        return []


def _build_universe() -> list[str]:
    sp500 = _fetch_sp500_tickers()
    combined = list(dict.fromkeys(sp500 + SUPPLEMENTAL))   # preserve order, deduplicate
    print(f"  Total unique tickers: {len(combined)}")
    return combined


def _download(tickers: list[str]) -> pd.DataFrame:
    """Download adjusted-close prices with per-ticker retry + progress."""
    series: dict[str, pd.Series] = {}
    failed: list[str] = []

    for i, ticker in enumerate(tickers, start=1):
        ser = None
        for attempt in range(3):
            try:
                hist = yf.download(
                    ticker, start=START, end=END,
                    auto_adjust=True, progress=False, threads=False,
                )
                if hist is None or hist.empty or "Close" not in hist.columns:
                    raise ValueError("empty or malformed response")
                raw = hist["Close"].dropna()
                if isinstance(raw, pd.DataFrame):
                    raw = raw.iloc[:, 0]
                if len(raw) < MIN_ROWS:
                    raise ValueError(f"only {len(raw)} rows")
                ser = raw
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    failed.append(f"{ticker}: {e}")
        if ser is not None:
            ser.name = ticker
            series[ticker] = ser
        time.sleep(0.3)
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)} processed  ({len(series)} succeeded)")

    if failed:
        print(f"  {len(failed)} tickers failed:")
        for msg in failed[:10]:
            print(f"    - {msg}")
        if len(failed) > 10:
            print(f"    … {len(failed) - 10} more")

    # Align to US business-day calendar
    bday_index = pd.bdate_range(
        start=min(s.index.tz_localize(None).min() if s.index.tz else s.index.min()
                  for s in series.values()),
        end=max(s.index.tz_localize(None).max() if s.index.tz else s.index.max()
                for s in series.values()),
    )
    prices = pd.DataFrame(index=bday_index)
    for ticker, ser in series.items():
        idx = ser.index.tz_localize(None) if ser.index.tz is not None else ser.index
        prices[ticker] = pd.Series(ser.values, index=idx).reindex(bday_index)

    # Drop tickers with >30% missing on the business-day grid
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.7))
    return prices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download even if cache is fresh")
    args = parser.parse_args()

    print("=" * 68)
    print("TimesFM fine-tuning universe builder")
    print(f"  Range : {START} → {END}")
    print(f"  Output: {OUT_PATH.name}")
    print("=" * 68)

    # Cache check
    if not args.force and OUT_PATH.exists():
        age = datetime.now() - datetime.fromtimestamp(OUT_PATH.stat().st_mtime)
        if age < timedelta(days=CACHE_DAYS):
            try:
                existing = pd.read_parquet(OUT_PATH)
                print(f"\nExisting parquet is {age.total_seconds()/3600:.1f}h old "
                      f"({len(existing.columns)} tickers, {len(existing)} rows). "
                      f"Use --force to re-download.")
                return
            except Exception:
                pass

    print("\n[1/3] Building ticker universe…")
    universe = _build_universe()

    print(f"\n[2/3] Downloading {len(universe)} tickers ({START} → {END})…")
    prices = _download(universe)
    print(f"\n  Aligned price matrix: {len(prices)} trading days × {len(prices.columns)} tickers")

    if len(prices.columns) < MIN_TICKERS:
        sys.exit(
            f"ERROR: only {len(prices.columns)} tickers survived alignment, "
            f"need ≥{MIN_TICKERS}.  Yahoo may be rate-limiting — wait 15 min and retry."
        )

    print("\n[3/3] Converting to log-prices and saving parquet…")
    # Drop any remaining rows where ALL tickers are NaN (e.g. global holidays)
    prices = prices.dropna(how="all")

    # Forward-fill within-series gaps ≤5 days (market closures, data holes)
    # then drop any tickers that still have NaNs at the start (IPO after START)
    prices = prices.fillna(method="ffill", limit=5)
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.85))

    # Convert to log-prices.  clip(lower=1e-6) guards against any zero-price
    # artefacts in yfinance data.
    log_prices = np.log(prices.clip(lower=1e-6))

    # Sanity check: no inf / all-NaN columns
    log_prices = log_prices.replace([np.inf, -np.inf], np.nan)
    log_prices = log_prices.dropna(axis=1, thresh=int(len(log_prices) * 0.85))

    log_prices.to_parquet(OUT_PATH)

    print(f"\n  Saved: {OUT_PATH}")
    print(f"  Shape: {log_prices.shape[0]} rows × {log_prices.shape[1]} tickers")
    print(f"  Size : {OUT_PATH.stat().st_size / 1e6:.1f} MB")
    print("\nNext step:")
    print("  python scripts/finetune_timesfm.py")


if __name__ == "__main__":
    main()
