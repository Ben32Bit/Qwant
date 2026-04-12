"""
Persistent SQLite price store.

Architecture:
  - Historical closes (before today) never change → stored permanently, never re-fetched
  - Today's row may change during market hours → TTL handled by caller
  - Acts as L2 cache behind the in-memory TTLCache in data_service.py

Table: daily_prices(ticker TEXT, date TEXT, close REAL, PRIMARY KEY(ticker, date))
"""

import sqlite3
import pandas as pd
import os
import logging
from pathlib import Path
from datetime import date, timedelta

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("PRICE_DB_PATH", "./data/prices.db")


def init_db() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                ticker  TEXT NOT NULL,
                date    TEXT NOT NULL,
                close   REAL NOT NULL,
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticker ON daily_prices(ticker, date)"
        )
    logger.info("Price store initialised at %s", DB_PATH)


def store_prices(prices_df: pd.DataFrame) -> int:
    """
    Upsert a prices DataFrame (DatetimeIndex rows, ticker columns) into SQLite.
    Returns the number of rows written.
    """
    rows = []
    for ticker in prices_df.columns:
        series = prices_df[ticker].dropna()
        for dt, close in series.items():
            rows.append((ticker, dt.strftime("%Y-%m-%d"), float(close)))

    if not rows:
        return 0

    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO daily_prices (ticker, date, close) VALUES (?,?,?)",
            rows,
        )
    logger.debug("Stored %d price rows", len(rows))
    return len(rows)


def load_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load prices from SQLite for the given tickers and date range.
    Returns a DataFrame with DatetimeIndex and tickers as columns.
    Returns an empty DataFrame if no data found.
    """
    if not tickers:
        return pd.DataFrame()

    placeholders = ",".join("?" * len(tickers))
    query = f"""
        SELECT ticker, date, close
        FROM daily_prices
        WHERE ticker IN ({placeholders})
          AND date >= ? AND date <= ?
        ORDER BY date
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(query, tickers + [start_date, end_date]).fetchall()
    except sqlite3.OperationalError:
        # DB doesn't exist yet — return empty
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ticker", "date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="ticker", values="close")
    pivot.columns.name = None
    return pivot


def get_coverage(tickers: list[str]) -> dict[str, tuple[str, str]]:
    """
    Returns {ticker: (min_date, max_date)} for tickers present in the DB.
    Tickers with no data are absent from the result dict.
    """
    if not tickers:
        return {}

    placeholders = ",".join("?" * len(tickers))
    query = f"""
        SELECT ticker, MIN(date), MAX(date)
        FROM daily_prices
        WHERE ticker IN ({placeholders})
        GROUP BY ticker
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(query, tickers).fetchall()
    except sqlite3.OperationalError:
        return {}

    return {row[0]: (row[1], row[2]) for row in rows}


def tickers_needing_fetch(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> tuple[list[str], str, str]:
    """
    Given a requested (tickers, start, end), determine which tickers need
    a yfinance fetch and what the minimal date range to fetch is.

    Returns:
        (tickers_to_fetch, fetch_start, fetch_end)

    Logic:
    - If a ticker is completely absent from the DB → must fetch full range
    - If a ticker's max_date is earlier than end_date → fetch from (max_date - 5d) to end_date
      (5 day overlap ensures we don't miss the most recent trading day)
    - If a ticker's min_date is later than start_date → must re-fetch from start_date
      (user is asking for more history than we have)
    """
    coverage = get_coverage(tickers)
    today = date.today().isoformat()
    effective_end = min(end_date, today)

    needs_fetch: set[str] = set()
    earliest_needed = effective_end  # will be pushed backward

    for ticker in tickers:
        if ticker not in coverage:
            needs_fetch.add(ticker)
            earliest_needed = min(earliest_needed, start_date)
            continue

        min_dt, max_dt = coverage[ticker]

        if min_dt > start_date:
            # We don't have early enough history
            needs_fetch.add(ticker)
            earliest_needed = min(earliest_needed, start_date)

        if max_dt < effective_end:
            # We're missing recent days (could just be today's close)
            needs_fetch.add(ticker)
            # Fetch from overlap point to avoid gaps
            overlap_start = (
                pd.to_datetime(max_dt) - timedelta(days=5)
            ).strftime("%Y-%m-%d")
            earliest_needed = min(earliest_needed, overlap_start)

    return list(needs_fetch), earliest_needed, effective_end
