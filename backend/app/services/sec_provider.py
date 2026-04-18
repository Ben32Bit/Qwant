"""
SECProvider — SEC EDGAR insider trade data (Form 4).

Fetches open-market insider transactions for portfolio tickers and aggregates
them into features that reflect management conviction about the company's
near-term prospects.

Research basis
--------------
Seyhun, H.N. (1986). Insiders' profits, costs of trading, and market efficiency.
  Journal of Financial Economics, 16(2), 189–212.
  https://doi.org/10.1016/0304-405X(86)90060-7
Lakonishok, J. & Lee, I. (2001). Are insider trades informative?
  Review of Financial Studies, 14(1), 79–111.
  https://doi.org/10.1093/rfs/14.1.79

API
---
SEC EDGAR REST API — no authentication required; free and public.
Fair Access Policy: 10 req/s max. Include User-Agent with contact.
https://www.sec.gov/os/accessing-edgar-data

Endpoints used:
  company_tickers.json  : ticker → CIK mapping (downloaded once)
  submissions/CIK{}.json : recent filings list
  Archives/edgar/data/  : Form 4 XML documents

Features returned (per ticker + portfolio aggregate)
-----------------------------------------------------
  net_buying_30d     : dollar-weighted net buying past 30 days (buys − sells)
                       open-market only (transaction codes P/S); positive = insiders buying
  buy_ratio_30d      : buys / (buys + sells) — 0 = pure selling, 1 = pure buying
  officer_signal     : same net buying but restricted to officers (higher conviction)
  director_signal    : same net buying but directors only
  n_transactions_30d : number of open-market transactions in window

Point-in-time: uses transactionDate (not filingDate) per academic best practice.

Caching
-------
  CIK lookups:   permanent in-process (tickers don't change CIK)
  Form 4 data:   24-hour in-process cache
  Feature dict:  24-hour in-process cache

ETFs / funds are automatically skipped (no insider trades).
Individual ticker failures are isolated — portfolio aggregate is computed on
available data only.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── EDGAR constants ───────────────────────────────────────────────────────────

# EDGAR Fair Access Policy requires User-Agent with contact info
_UA = "QwantPortfolioBacktester admin@qwant.app"
# No Accept-Encoding — urllib.request doesn't auto-decompress; keep responses plain
_HEADERS = {"User-Agent": _UA}

EDGAR_API   = "https://data.sec.gov"
EDGAR_FILES = "https://www.sec.gov/Archives/edgar/data"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_RATE_SLEEP   = 0.12   # seconds between requests (max 10/s)
CACHE_TTL     = 86_400 # 24 hours
LOOKBACK_DAYS = 30

# ETF / fund / index tickers — no insider trades exist
_SKIP_TICKERS = {
    "SPY", "QQQ", "IWM", "DIA", "MDY", "IJH", "IJR",
    "EFA", "EEM", "VEA", "VWO", "IEFA",
    "TLT", "IEF", "SHY", "SHV", "BIL", "AGG", "LQD", "HYG", "JNK", "BND",
    "GLD", "SLV", "USO", "UNG", "PDBC",
    "VTI", "VOO", "IVV", "SCHB", "ITOT",
    "VNQ", "XLRE",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLB", "XLP", "XLU", "XLC",
    "ARKK", "ARKQ", "ARKG",
}

# Open-market transaction codes (exclude grants, awards, option exercises)
_OPEN_MARKET = {"P", "S"}   # P = open market purchase, S = open market sale

# ── In-process caches ─────────────────────────────────────────────────────────

_cik_map:      dict[str, str | None]          = {}   # ticker → "0000320193" | None
_form4_cache:  dict[str, tuple[float, list]]  = {}   # cik → (ts, [transactions])
_feature_cache: dict[str, tuple[float, dict]] = {}   # date → (ts, features)


# ── Public interface ──────────────────────────────────────────────────────────

def get_insider_features(
    tickers: list[str],
    weights: dict[str, float],
    as_of_date: Optional[str] = None,
) -> dict:
    """
    Return insider trade features for a portfolio.

    Parameters
    ----------
    tickers  : list of ticker symbols in the portfolio
    weights  : {ticker: weight} — used to weight portfolio-level aggregate
    as_of_date : ISO date string (YYYY-MM-DD). Defaults to today.

    Returns
    -------
    dict with:
        available          : bool
        portfolio_net_buying_30d   : float  (positive = net buying; $M weighted by portfolio weight)
        portfolio_buy_ratio_30d    : float  (0–1)
        officer_signal_30d         : float
        n_transactions_30d         : int
        per_ticker         : {ticker: {net_buying_30d, buy_ratio_30d, n_transactions}}
        source             : "sec_edgar"
    """
    key = (as_of_date or date.today().isoformat()) + "|" + ",".join(sorted(tickers))

    if key in _feature_cache:
        ts, result = _feature_cache[key]
        if time.time() - ts < CACHE_TTL:
            return result

    result = _compute_insider_features(tickers, weights, as_of_date or date.today().isoformat())
    _feature_cache[key] = (time.time(), result)
    return result


# ── Internal computation ──────────────────────────────────────────────────────

def _compute_insider_features(tickers, weights, as_of_date: str) -> dict:
    eligible = [t for t in tickers if t not in _SKIP_TICKERS]
    if not eligible:
        return _neutral_result("no_eligible_tickers")

    cutoff = (date.fromisoformat(as_of_date) - timedelta(days=LOOKBACK_DAYS)).isoformat()

    per_ticker: dict[str, dict] = {}
    for ticker in eligible:
        try:
            cik = _get_cik(ticker)
            if not cik:
                continue
            transactions = _get_form4_transactions(cik, ticker, cutoff, as_of_date)
            per_ticker[ticker] = _aggregate_transactions(transactions)
        except Exception as exc:
            logger.debug("SECProvider: %s failed (%s) — skipping", ticker, exc)

    if not per_ticker:
        return _neutral_result("all_tickers_failed")

    # Portfolio-weighted aggregate
    total_weight  = sum(abs(weights.get(t, 0.0)) for t in per_ticker)
    if total_weight < 1e-9:
        total_weight = len(per_ticker)

    port_net_buying  = 0.0
    port_buy_ratio   = 0.0
    port_officer_sig = 0.0
    n_txn_total      = 0
    for ticker, feats in per_ticker.items():
        w = abs(weights.get(ticker, 1.0)) / total_weight
        port_net_buying  += w * feats["net_buying_30d"]
        port_buy_ratio   += w * feats["buy_ratio_30d"]
        port_officer_sig += w * feats["officer_signal_30d"]
        n_txn_total      += feats["n_transactions_30d"]

    return {
        "available":                    True,
        "portfolio_net_buying_30d":     round(port_net_buying,  2),
        "portfolio_buy_ratio_30d":      round(port_buy_ratio,   3),
        "officer_signal_30d":           round(port_officer_sig, 2),
        "n_transactions_30d":           n_txn_total,
        "n_tickers_with_data":          len(per_ticker),
        "per_ticker":                   per_ticker,
        "source":                       "sec_edgar",
    }


def _aggregate_transactions(transactions: list[dict]) -> dict:
    """Compute features from a list of Form 4 open-market transactions."""
    bought_m  = 0.0   # $ millions, bought
    sold_m    = 0.0   # $ millions, sold
    off_buy   = 0.0
    off_sell  = 0.0
    n_buy = n_sell = 0

    for txn in transactions:
        code   = txn.get("code", "")
        if code not in _OPEN_MARKET:
            continue
        ad     = txn.get("acquired_disposed", "")
        shares = txn.get("shares", 0.0)
        price  = txn.get("price", 0.0)
        is_off = txn.get("is_officer", False)
        dollars_m = shares * price / 1_000_000

        if ad == "A":
            bought_m += dollars_m
            n_buy    += 1
            if is_off:
                off_buy += dollars_m
        elif ad == "D":
            sold_m  += dollars_m
            n_sell  += 1
            if is_off:
                off_sell += dollars_m

    net_buying   = bought_m - sold_m
    n_total      = n_buy + n_sell
    buy_ratio    = n_buy / n_total if n_total > 0 else 0.5
    officer_sig  = off_buy - off_sell

    return {
        "net_buying_30d":    round(net_buying,  2),
        "buy_ratio_30d":     round(buy_ratio,   3),
        "officer_signal_30d": round(officer_sig, 2),
        "n_transactions_30d": n_total,
    }


def _neutral_result(reason: str) -> dict:
    return {
        "available":                    False,
        "portfolio_net_buying_30d":     0.0,
        "portfolio_buy_ratio_30d":      0.5,
        "officer_signal_30d":           0.0,
        "n_transactions_30d":           0,
        "n_tickers_with_data":          0,
        "per_ticker":                   {},
        "source":                       reason,
    }


# ── CIK lookup ────────────────────────────────────────────────────────────────

_cik_full_map: dict[str, str] | None = None   # ticker_upper → "0000320193"


def _load_cik_map() -> dict[str, str]:
    global _cik_full_map
    if _cik_full_map is not None:
        return _cik_full_map

    try:
        data = _fetch_json(TICKERS_URL)
        mapping: dict[str, str] = {}
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).upper()
            cik    = str(entry.get("cik_str", "")).zfill(10)
            if ticker:
                mapping[ticker] = cik
        _cik_full_map = mapping
        logger.info("SECProvider: loaded %d ticker→CIK mappings", len(mapping))
        return mapping
    except Exception as exc:
        logger.warning("SECProvider: CIK map load failed (%s)", exc)
        _cik_full_map = {}
        return {}


def _get_cik(ticker: str) -> str | None:
    t = ticker.upper()
    if t in _cik_map:
        return _cik_map[t]
    mapping = _load_cik_map()
    cik = mapping.get(t)
    _cik_map[t] = cik
    return cik


# ── Form 4 fetching & parsing ─────────────────────────────────────────────────

def _get_form4_transactions(cik: str, ticker: str, cutoff: str, end: str) -> list[dict]:
    """Return open-market Form 4 transactions for [cutoff, end]."""
    ts_key = cik
    if ts_key in _form4_cache:
        ts, cached = _form4_cache[ts_key]
        if time.time() - ts < CACHE_TTL:
            return [t for t in cached if cutoff <= t.get("date", "") <= end]

    # Fetch submissions
    url  = f"{EDGAR_API}/submissions/CIK{cik}.json"
    subs = _fetch_json(url)
    time.sleep(_RATE_SLEEP)

    recent = subs.get("filings", {}).get("recent", {})
    forms   = recent.get("form", [])
    dates   = recent.get("filingDate", [])
    acc_nos = recent.get("accessionNumber", [])
    pri_doc = recent.get("primaryDocument", [])

    # Collect Form 4 filing info within a generous window (last 90 days for cache)
    ninety_ago = (date.today() - timedelta(days=90)).isoformat()
    filings_to_fetch = []
    for i, form in enumerate(forms):
        if form not in ("4", "4/A"):
            continue
        if i >= len(dates) or dates[i] < ninety_ago:
            continue
        if i >= len(acc_nos):
            continue
        acc = acc_nos[i].replace("-", "")
        doc = pri_doc[i] if i < len(pri_doc) else ""
        filings_to_fetch.append({"acc": acc, "doc": doc, "filing_date": dates[i]})

    # Parse each XML
    all_transactions: list[dict] = []
    for filing in filings_to_fetch[:20]:   # cap at 20 recent Form 4s
        try:
            txns = _fetch_and_parse_form4(cik, filing["acc"], filing["doc"])
            all_transactions.extend(txns)
            time.sleep(_RATE_SLEEP)
        except Exception as exc:
            logger.debug("SECProvider: %s Form4 parse error (%s)", ticker, exc)

    _form4_cache[ts_key] = (time.time(), all_transactions)
    return [t for t in all_transactions if cutoff <= t.get("date", "") <= end]


def _fetch_and_parse_form4(cik: str, acc_nodash: str, primary_doc: str) -> list[dict]:
    """Download one Form 4 XML and extract transactions."""
    cik_int = int(cik)

    # primaryDocument may have an XSLT stylesheet prefix like "xslF345X06/form4.xml".
    # That prefix is a rendering hint only — the actual file lives at the archive root.
    filename = primary_doc.split("/")[-1] if "/" in primary_doc else primary_doc
    if not filename.lower().endswith(".xml"):
        filename = "form4.xml"   # near-universal Form 4 filename

    url = f"{EDGAR_FILES}/{cik_int}/{acc_nodash}/{filename}"
    try:
        xml_text = _fetch_text(url)
    except Exception:
        # Fallback: canonical accession-number filename (e.g. 0001234567-24-000001.xml)
        alt = f"{acc_nodash[:10]}-{acc_nodash[10:12]}-{acc_nodash[12:]}.xml"
        xml_text = _fetch_text(f"{EDGAR_FILES}/{cik_int}/{acc_nodash}/{alt}")

    return _parse_form4_xml(xml_text)


def _parse_form4_xml(xml_text: str) -> list[dict]:
    """Parse Form 4 XML. Returns list of transaction dicts."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    # Reporter role
    rel = root.find(".//reportingOwnerRelationship")
    is_director = _xml_bool(rel, "isDirector")
    is_officer  = _xml_bool(rel, "isOfficer")
    is_10pct    = _xml_bool(rel, "isTenPercentOwner")

    transactions: list[dict] = []

    # Non-derivative (stock) transactions
    for txn_el in root.findall(".//nonDerivativeTransaction"):
        txn = _parse_transaction_element(txn_el)
        if txn:
            txn.update({
                "is_director": is_director,
                "is_officer":  is_officer,
                "is_10pct":    is_10pct,
            })
            transactions.append(txn)

    return transactions


def _parse_transaction_element(el: ET.Element) -> dict | None:
    """Extract fields from a <nonDerivativeTransaction> element."""
    try:
        txn_date = _xml_val(el, ".//transactionDate/value")
        code     = _xml_val(el, ".//transactionCoding/transactionCode")
        shares   = float(_xml_val(el, ".//transactionShares/value") or "0")
        price    = float(_xml_val(el, ".//transactionPricePerShare/value") or "0")
        ad_code  = _xml_val(el, ".//transactionAcquiredDisposedCode/value")

        if not txn_date or not code:
            return None

        return {
            "date":              txn_date,
            "code":              code,
            "shares":            shares,
            "price":             price,
            "acquired_disposed": ad_code or "",
        }
    except (ValueError, TypeError):
        return None


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())


def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read()
    # Try decode as UTF-8, fall back to latin-1
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _xml_val(el: ET.Element, path: str) -> str:
    found = el.find(path)
    return (found.text or "").strip() if found is not None else ""


def _xml_bool(el: ET.Element | None, tag: str) -> bool:
    if el is None:
        return False
    found = el.find(tag)
    return bool(found is not None and found.text and found.text.strip() == "1")
