from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.models.portfolio import PortfolioInput
from app.models.backtest_result import BacktestResult
from app.services.backtest_engine import run_full_backtest
from app.services.data_service import fetch_prices
from app.services.optimization import apply_strategy
import os

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))


@router.post("/backtest", response_model=BacktestResult)
@limiter.limit("60/hour")
async def run_backtest(request: Request, portfolio: PortfolioInput) -> BacktestResult:
    """
    Direct backtest endpoint — bypasses AI.
    Used for manual portfolio builder in the UI.
    """
    try:
        if portfolio.strategy != "custom":
            tickers = [a.ticker for a in portfolio.assets]
            prices = fetch_prices(
                list(set(tickers + [portfolio.benchmark])),
                portfolio.start_date,
                portfolio.end_date,
            )
            port_prices = prices[[t for t in tickers if t in prices.columns]]
            returns_df = port_prices.pct_change().dropna()
            current_weights = {a.ticker: a.weight for a in portfolio.assets}
            optimized = apply_strategy(portfolio.strategy, returns_df, current_weights, RISK_FREE_RATE)
            for asset in portfolio.assets:
                if asset.ticker in optimized:
                    asset.weight = float(optimized[asset.ticker])

        result = run_full_backtest(portfolio)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")
