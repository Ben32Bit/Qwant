import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

load_dotenv()

from app.routers import chat, backtest, data, screen
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


# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initialising price store…")
    init_db()
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


@app.get("/health")
async def health():
    return {"status": "ok"}
