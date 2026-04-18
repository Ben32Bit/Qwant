"""
Shared slowapi rate limiter — single instance used by all routers.

Limits (all per-IP, in-memory):
  AI_LIMITS    — endpoints that call Anthropic (chat, screen/chat, unified/chat)
  FCST_LIMITS  — /forecast  (heavy CPU, no Anthropic)
  CMPT_LIMITS  — /backtest, /screen/run, /screen/backtest, /regime/current
  GLOBAL_LIMIT — default cap applied to every route that uses this limiter

All values are configurable via environment variables.
slowapi 0.1.9 requires plain strings ("N/period"), not lists.
"""
import os
from slowapi import Limiter
from slowapi.util import get_remote_address

# ── Configurable limits ───────────────────────────────────────────────────────
# AI endpoints — each call may cost $0.003–0.01 in Anthropic tokens.
# 30/hour ≈ one request every 2 min — fine for human use, blocks bots.
AI_LIMITS   = os.getenv("AI_RATE_LIMIT",       "30/hour")

# Forecast — 2 server calls per full run (phase 1 + 2) → 10 full runs/hour.
FCST_LIMITS = os.getenv("FORECAST_RATE_LIMIT", "20/hour")

# Compute-only endpoints — no AI cost, just CPU.
CMPT_LIMITS = os.getenv("COMPUTE_RATE_LIMIT",  "60/hour")

# Global safety net — applied to every route via default_limits.
GLOBAL_LIMIT = os.getenv("GLOBAL_RATE_LIMIT",  "300/hour")

# ── Shared limiter instance ───────────────────────────────────────────────────
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[GLOBAL_LIMIT],
    headers_enabled=True,   # sends X-RateLimit-Limit / X-RateLimit-Remaining headers
)
