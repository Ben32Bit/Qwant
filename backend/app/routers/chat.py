from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest, ChatResponse, PortfolioResponse
from app.services.ai_service import call_ai
from app.services.backtest_engine import run_full_backtest
from app.services.data_service import fetch_prices
from app.services.optimization import apply_strategy
import os

router = APIRouter()
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main conversational endpoint.
    Accepts natural language, uses AI to construct a portfolio, runs backtest.
    """
    try:
        # Step 1: AI constructs portfolio from natural language
        portfolio_input, ai_text, strategy_summary = call_ai(request)

        # Step 2: Apply optimization if strategy is not custom
        if portfolio_input.strategy != "custom":
            tickers = [a.ticker for a in portfolio_input.assets]
            prices = fetch_prices(
                list(set(tickers + [portfolio_input.benchmark])),
                portfolio_input.start_date,
                portfolio_input.end_date,
            )
            port_prices = prices[[t for t in tickers if t in prices.columns]]
            returns_df = port_prices.pct_change().dropna()
            current_weights = {a.ticker: a.weight for a in portfolio_input.assets}
            optimized = apply_strategy(
                portfolio_input.strategy, returns_df, current_weights, RISK_FREE_RATE
            )
            for asset in portfolio_input.assets:
                if asset.ticker in optimized:
                    asset.weight = float(optimized[asset.ticker])

        # Step 3: Run backtest
        backtest_result = run_full_backtest(portfolio_input)

        # Step 4: Build response
        portfolio_response = PortfolioResponse(
            assets=portfolio_input.assets,
            strategy=portfolio_input.strategy,
            rebalance_frequency=portfolio_input.rebalance_frequency,
            benchmark=portfolio_input.benchmark,
            start_date=portfolio_input.start_date,
            end_date=portfolio_input.end_date,
            strategy_summary=strategy_summary,
        )

        return ChatResponse(
            portfolio=portfolio_response,
            backtest=backtest_result,
            ai_response=ai_text,
            strategy_summary=strategy_summary,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat request failed: {str(e)}")
