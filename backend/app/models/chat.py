from pydantic import BaseModel
from typing import Optional
from app.models.portfolio import AssetInput, PortfolioInput
from app.models.backtest_result import BacktestResult


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_history: list[ChatMessage] = []
    current_portfolio: Optional[PortfolioInput] = None


class PortfolioResponse(BaseModel):
    assets: list[AssetInput]
    strategy: str
    rebalance_frequency: str
    benchmark: str
    start_date: str
    end_date: str
    strategy_summary: Optional[str] = None


class DisplayConfig(BaseModel):
    sections: list[str] = ["equity_curve", "drawdown", "metrics_summary"]
    featured_metrics: list[str] = ["cagr", "sharpe", "max_drawdown", "volatility"]
    narrative: Optional[str] = None


class ChatResponse(BaseModel):
    portfolio: PortfolioResponse
    backtest: BacktestResult
    ai_response: str
    strategy_summary: Optional[str] = None
    display_config: DisplayConfig = DisplayConfig()
