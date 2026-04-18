"""
Shared slowapi rate limiter — single instance used by all routers.

Limits (all per-IP, in-memory):
  AI_LIMITS    — endpoints that call Anthropic (chat, screen/chat, unified/chat)
  FCST_LIMITS  — /forecast  (heavy CPU, no Anthropic)
  CMPT_LIMITS  — /backtest, /screen/run, /screen/backtest, /regime/current
  GLOBAL_LIMIT — default cap applied to every route that uses this limiter

All values are configurable via environment variables.
"""
import os
from slowapi import Limiter
from slowapi.util import get_remote_address

# ── Configurable limits ───────────────────────────────────────────────────────
# AI endpoints — each call may cost $0.003–0.01 in Anthropic tokens.
# 10/hour   ≈ comfortable for a human, painful for a bot
# 50/day    ≈ $0.25–0.50 max spend per IP per day
AI_LIMITS   = os.getenv("AI_RATE_LIMIT",       "10/hour;50/day").split(";")

# Forecast — two server calls per full run (phase 1 + phase 2); no Anthropic cost
# 12/hour   ≈ 6 full forecast runs / hour per IP
FCST_LIMITS = os.getenv("FORECAST_RATE_LIMIT", "12/hour;40/day").split(";")

# Compute-only endpoints — no AI cost, just CPU
CMPT_LIMITS = os.getenv("COMPUTE_RATE_LIMIT",  "30/hour;120/day").split(";")

# Global safety net — applied to every route via default_limits
GLOBAL_LIMIT = os.getenv("GLOBAL_RATE_LIMIT",  "200/hour")

# ── Shared limiter instance ───────────────────────────────────────────────────
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[GLOBAL_LIMIT],
    headers_enabled=True,   # sends X-RateLimit-Limit / X-RateLimit-Remaining headers
)
