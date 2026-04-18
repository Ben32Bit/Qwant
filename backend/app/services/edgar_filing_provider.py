"""
EDGARFilingProvider — SEC 10-K/10-Q excerpt extraction for filing sentiment.

Fetches the most recent 10-K or 10-Q for each portfolio company and extracts
key text excerpts (Risk Factors, MD&A) for browser-side FinBERT scoring.

Two-path strategy
-----------------
1. EDGAR EFTS full-text search (fast, returns highlighted snippets, no large
   document download needed).
2. Fallback: submissions API + primary document fetch + regex section extraction.

Both paths are cached 24h per ticker.

API usage
---------
EDGAR EFTS: https://efts.sec.gov/LATEST/search-index
  No authentication. EDGAR Fair Access Policy: max 10 req/s.
  User-Agent with contact email required.

Tickers mapped to CIK via https://www.sec.gov/files/company_tickers.json
(same source as sec_provider.py; cached permanently).

References
----------
Loughran, T. & McDonald, B. (2011). When is a Liability not a Liability?
  Journal of Finance, 66(1), 35–65. doi:10.1111/j.1540-6261.2010.01625.x
  (Linguistic cues in 10-K filings predict future returns)
Tetlock, P.C. (2007). Giving Content to Investor Sentiment: The Role of
  Media in the Stock Market. Journal of Finance, 62(3), 1139–1168.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_UA      = "QwantPortfolioBacktester admin@qwant.app"
_HEADERS = {"User-Agent": _UA, "Accept": "application/json"}

EDGAR_API   = "https://data.sec.gov"
EDGAR_FILES = "https://www.sec.gov/Archives/edgar/data"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EFTS_URL    = "https://efts.sec.gov/LATEST/search-index"

_RATE_SLEEP = 0.15   # 10 req/s max per EDGAR policy
CACHE_TTL   = 86_400  # 24h

# ETFs/funds — skip (file no 10-K as an operating company)
_SKIP_TICKERS = {
    "SPY","QQQ","IWM","DIA","MDY","IJH","IJR",
    "EFA","EEM","VEA","VWO","IEFA",
    "TLT","IEF","SHY","SHV","BIL","AGG","LQD","HYG","JNK","BND",
    "GLD","SLV","USO","UNG","PDBC",
    "VTI","VOO","IVV","SCHB","ITOT",
    "VNQ","XLRE",
    "XLK","XLF","XLE","XLV","XLY","XLI","XLB","XLP","XLU","XLC",
    "ARKK","ARKQ","ARKG","VIXY","VXX","SVXY",
}

# ── In-process caches ─────────────────────────────────────────────────────────

_meta_map: dict[str, dict] | None = None        # ticker → {cik, name}
_filing_cache: dict[str, tuple[float, dict]] = {}  # ticker|date → (ts, result)


# ── Ticker → CIK + company name ───────────────────────────────────────────────

def _load_ticker_meta() -> dict[str, dict]:
    global _meta_map
    if _meta_map is not None:
        return _meta_map
    try:
        req = urllib.request.Request(TICKERS_URL, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        mapping: dict[str, dict] = {}
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).upper()
            cik    = str(entry.get("cik_str", "")).zfill(10)
            name   = str(entry.get("title", ""))
            if ticker and cik:
                mapping[ticker] = {"cik": cik, "name": name}
        _meta_map = mapping
        logger.info("EDGARFiling: loaded %d ticker→CIK mappings", len(mapping))
        return mapping
    except Exception as exc:
        logger.warning("EDGARFiling: CIK meta load failed: %s", exc)
        _meta_map = {}
        return {}


def _get_meta(ticker: str) -> dict | None:
    return _load_ticker_meta().get(ticker.upper())


# ── Path 1: EFTS full-text search ─────────────────────────────────────────────

def _efts_snippets(entity_name: str, start_date: str, end_date: str) -> list[dict]:
    """
    Search EDGAR EFTS for recent 10-K/10-Q excerpts mentioning risk/outlook language.
    Returns list of {form, filing_date, snippets: [str]}.
    """
    from urllib.parse import quote

    # Broad risk/outlook query — ensures we get relevant excerpt highlights
    q = quote('"risk factors" OR "forward-looking" OR "material uncertainty"')
    entity = quote(entity_name)

    url = (
        f"{EFTS_URL}?"
        f"q={q}"
        f"&entity={entity}"
        f"&forms=10-K%2C10-Q"
        f"&dateRange=custom"
        f"&startdt={start_date}"
        f"&enddt={end_date}"
    )

    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        logger.debug("EDGARFiling EFTS failed (%s): %s", entity_name, exc)
        return []

    hits = data.get("hits", {}).get("hits", [])
    out = []
    for hit in hits[:2]:    # cap at 2 most recent filings
        source     = hit.get("_source", {})
        highlights = hit.get("highlight", {}).get("file_contents", [])
        snippets = []
        for hl in highlights[:5]:
            clean = re.sub(r'<[^>]+>', '', hl).strip()
            clean = re.sub(r'\s+', ' ', clean)
            if len(clean) > 60:
                snippets.append(clean[:700])
        if snippets:
            out.append({
                "form":        source.get("form_type", "10-K"),
                "filing_date": source.get("file_date", ""),
                "snippets":    snippets,
            })

    return out


# ── Path 2: Submissions API + primary document fallback ───────────────────────

def _recent_filing_meta(cik: str) -> dict | None:
    """Get most-recent 10-K or 10-Q accession metadata from submissions JSON."""
    url = f"{EDGAR_API}/submissions/CIK{cik}.json"
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            subs = json.loads(resp.read())
    except Exception as exc:
        logger.debug("EDGARFiling: submissions fetch failed for CIK %s: %s", cik, exc)
        return None

    recent  = subs.get("filings", {}).get("recent", {})
    forms   = recent.get("form", [])
    dates   = recent.get("filingDate", [])
    acc_nos = recent.get("accessionNumber", [])
    pri_doc = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
            continue
        return {
            "form":        form,
            "filing_date": dates[i]   if i < len(dates)   else "",
            "acc_nodash":  acc_nos[i].replace("-", "") if i < len(acc_nos) else "",
            "primary_doc": pri_doc[i] if i < len(pri_doc) else "",
        }
    return None


def _fetch_filing_excerpt(cik: str, acc_nodash: str, primary_doc: str) -> list[str]:
    """
    Download primary filing document (HTML), strip tags, extract Risk Factors
    and MD&A section openings. Returns list of text excerpts.
    """
    cik_int  = int(cik)
    filename = primary_doc.split("/")[-1] if "/" in primary_doc else primary_doc
    if not filename.lower().endswith((".htm", ".html")):
        filename = f"{acc_nodash}.htm"

    url = f"{EDGAR_FILES}/{cik_int}/{acc_nodash}/{filename}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=25) as resp:
            raw = resp.read(307_200)   # 300 KB cap
        try:
            html = raw.decode("utf-8", errors="replace")
        except Exception:
            html = raw.decode("latin-1", errors="replace")
    except Exception as exc:
        logger.debug("EDGARFiling: document fetch failed: %s", exc)
        return []

    # Strip HTML
    text = re.sub(r'<[^>]{1,400}>', ' ', html)
    for entity, repl in (('&nbsp;', ' '), ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('&#160;', ' ')):
        text = text.replace(entity, repl)
    text = re.sub(r'\s+', ' ', text)

    excerpts = []
    # Risk Factors (Item 1A)
    m = re.search(r'\bITEM\s+1A\b[.\s\-–]*RISK\s+FACTORS', text, re.IGNORECASE)
    if m:
        start = m.end()
        end_m = re.search(r'\bITEM\s+[2-9][AB]?\b', text[start:start + 15_000], re.IGNORECASE)
        section = text[start: start + (end_m.start() if end_m else 3_000)].strip()
        if len(section) > 100:
            excerpts.append(section[:1_200])

    # MD&A (Item 7)
    m = re.search(r'\bITEM\s+7\b[.\s\-–]*MANAGEMENT.{0,80}DISCUSSION', text, re.IGNORECASE)
    if m:
        start = m.end()
        end_m = re.search(r'\bITEM\s+[89][AB]?\b', text[start:start + 15_000], re.IGNORECASE)
        section = text[start: start + (end_m.start() if end_m else 3_000)].strip()
        if len(section) > 100:
            excerpts.append(section[:1_200])

    return excerpts


def _fallback_snippets(cik: str) -> list[dict]:
    filing = _recent_filing_meta(cik)
    if not filing:
        return []
    time.sleep(_RATE_SLEEP)
    excerpts = _fetch_filing_excerpt(cik, filing["acc_nodash"], filing["primary_doc"])
    if not excerpts:
        return []
    return [{"form": filing["form"], "filing_date": filing["filing_date"], "snippets": excerpts}]


# ── Public interface ──────────────────────────────────────────────────────────

def get_edgar_filing_context(
    tickers: list[str],
    weights: dict[str, float],
    as_of_date: Optional[str] = None,
) -> dict:
    """
    Return recent 10-K/10-Q excerpts for portfolio companies.

    Parameters
    ----------
    tickers     : portfolio ticker list
    weights     : {ticker: weight} — top-5 by |weight| are processed
    as_of_date  : ISO date string; defaults to today

    Returns
    -------
    {
        "per_ticker": {ticker: {form, filing_date, company, excerpts: [str], source}},
        "tickers_with_data": int,
        "available": bool,
    }
    """
    today   = as_of_date or date.today().isoformat()
    start1y = (date.fromisoformat(today) - timedelta(days=400)).isoformat()  # ~13 months

    eligible = [t for t in tickers if t not in _SKIP_TICKERS]
    eligible = sorted(eligible, key=lambda t: -abs(weights.get(t, 0)))[:5]

    per_ticker: dict[str, dict] = {}

    for ticker in eligible:
        cache_key = f"{ticker}|{today[:7]}"   # monthly granularity (filings don't change hourly)
        if cache_key in _filing_cache:
            ts, cached = _filing_cache[cache_key]
            if time.time() - ts < CACHE_TTL:
                if cached:
                    per_ticker[ticker] = cached
                continue

        meta = _get_meta(ticker)
        if not meta:
            _filing_cache[cache_key] = (time.time(), {})
            continue

        result: dict = {}

        # Path 1: EFTS
        try:
            filings = _efts_snippets(meta["name"], start1y, today)
            time.sleep(_RATE_SLEEP)
            if filings:
                f = filings[0]
                result = {
                    "form":        f["form"],
                    "filing_date": f["filing_date"],
                    "company":     meta["name"],
                    "excerpts":    f["snippets"],
                    "source":      "efts",
                }
        except Exception as exc:
            logger.debug("EDGARFiling EFTS path failed for %s: %s", ticker, exc)

        # Path 2: fallback via submissions + primary doc
        if not result:
            try:
                filings = _fallback_snippets(meta["cik"])
                time.sleep(_RATE_SLEEP)
                if filings:
                    f = filings[0]
                    result = {
                        "form":        f["form"],
                        "filing_date": f["filing_date"],
                        "company":     meta["name"],
                        "excerpts":    f["snippets"],
                        "source":      "primary_doc",
                    }
            except Exception as exc:
                logger.debug("EDGARFiling fallback path failed for %s: %s", ticker, exc)

        _filing_cache[cache_key] = (time.time(), result)
        if result:
            per_ticker[ticker] = result

    return {
        "per_ticker":        per_ticker,
        "tickers_with_data": len(per_ticker),
        "available":         bool(per_ticker),
    }
