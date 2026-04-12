from pydantic import BaseModel, field_validator
from typing import Optional
from app.utils.constants import REBALANCE_FREQUENCIES, STRATEGY_TYPES, DEFAULT_BENCHMARK, DEFAULT_REBALANCE_FREQUENCY, DEFAULT_STRATEGY


class AssetInput(BaseModel):
    ticker: str
    weight: float
    rationale: Optional[str] = None

    @field_validator("ticker")
    @classmethod
    def ticker_upper(cls, v: str) -> str:
        return v.strip().upper()


class PortfolioInput(BaseModel):
    assets: list[AssetInput]
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    rebalance_frequency: str = DEFAULT_REBALANCE_FREQUENCY
    benchmark: str = DEFAULT_BENCHMARK
    initial_capital: float = 10_000.0
    strategy: str = DEFAULT_STRATEGY

    @field_validator("rebalance_frequency")
    @classmethod
    def valid_rebalance(cls, v: str) -> str:
        if v not in REBALANCE_FREQUENCIES:
            raise ValueError(f"rebalance_frequency must be one of {REBALANCE_FREQUENCIES}")
        return v

    @field_validator("strategy")
    @classmethod
    def valid_strategy(cls, v: str) -> str:
        if v not in STRATEGY_TYPES:
            raise ValueError(f"strategy must be one of {STRATEGY_TYPES}")
        return v
