import os
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import date
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

load_dotenv()

from app.routers import chat, backtest, data, screen, unified, forecast
from app.services.price_store import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Rate limiter (in-memory, per IP) ─────────────────────────────────────────
# Uses slowapi — no Redis needed for single-instance deployment.
# Switch storage_uri to "redis://..." when running multiple workers.
CHAT_RATE    = os.getenv("CHAT_RATE_LIMIT",    "20/hour")
BACKTEST_RATE = os.getenv("BACKTEST_RATE_LIMIT", "60/hour")

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],          # no global limit; set per-route
    headers_enabled=True,       # sends X-RateLimit-* headers
)


# ── Warm universe ─────────────────────────────────────────────────────────────
# Top 20 ETFs by AUM + top 20 US stocks by average daily volume.
# Pre-fetched on startup so screener/portfolio requests hit SQLite instead of yfinance.
WARM_UNIVERSE = [
    # Core ETFs
    "SPY", "QQQ", "IWM", "VTI", "AGG", "BND", "TLT", "IEF", "SHY",
    "GLD", "SLV", "VNQ", "EFA", "EEM", "VEA", "VWO", "HYG", "LQD",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "XLB", "XLI", "XLRE", "XLC",
    "DBC", "USO", "VIXY",
    # High-volume US stocks
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO",
    "NFLX", "AMD", "ORCL", "CRM", "ADBE", "INTC", "QCOM",
    "JPM", "BAC", "GS", "V", "MA",
]


async def _warm_price_cache() -> None:
    """
    Background task: fetch 10 years of daily prices for the warm universe into SQLite.
    Runs once on startup; subsequent restarts are nearly free (SQLite L2 hit).
    Estimated storage: ~8 MB total for 50 tickers × 2,520 trading days.
    """
    from app.services.data_service import fetch_prices

    end_str = date.today().strftime("%Y-%m-%d")
    start_str = date(date.today().year - 10, date.today().month, date.today().day).strftime("%Y-%m-%d")

    logger.info("Warm-cache: fetching %d tickers from %s → %s", len(WARM_UNIVERSE), start_str, end_str)
    try:
        # Run in a thread so we don't block the event loop
        await asyncio.to_thread(fetch_prices, WARM_UNIVERSE, start_str, end_str)
        logger.info("Warm-cache: complete.")
    except Exception as exc:
        # Non-fatal — the app still works, just without the warm cache
        logger.warning("Warm-cache: failed (non-fatal): %s", exc)


# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initialising price store…")
    init_db()
    # Fire-and-forget: warm the price cache in the background.
    # Does not block startup; app serves requests while cache fills.
    asyncio.create_task(_warm_price_cache())
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Qwant Portfolio Backtester API",
    description="AI-powered portfolio construction and backtesting",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Make limiter + rate strings available to routers via app.state
app.state.chat_rate    = CHAT_RATE
app.state.backtest_rate = BACKTEST_RATE

# Routers
app.include_router(chat.router,     prefix="/api")
app.include_router(backtest.router, prefix="/api")
app.include_router(data.router,     prefix="/api")
app.include_router(screen.router,   prefix="/api")
app.include_router(unified.router,  prefix="/api")
app.include_router(forecast.router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/debug/memory")
async def debug_memory():
    """Returns current process RSS memory in MB — used for RAM baseline auditing."""
    import psutil
    proc = psutil.Process()
    mem  = proc.memory_info()
    return {
        "rss_mb":  round(mem.rss  / 1024 / 1024, 1),
        "vms_mb":  round(mem.vms  / 1024 / 1024, 1),
        "pid":     proc.pid,
    }
