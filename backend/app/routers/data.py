from fastapi import APIRouter
from app.services.data_service import search_tickers
from app.utils.constants import BENCHMARK_TICKERS

router = APIRouter()


@router.get("/tickers/search")
async def ticker_search(q: str) -> list[dict]:
    """Autocomplete ticker search."""
    if not q or len(q) < 1:
        return []
    return search_tickers(q)


@router.get("/benchmarks")
async def benchmarks() -> list[dict]:
    """Return list of available benchmark ETFs."""
    return [
        {"ticker": ticker, "name": name}
        for ticker, name in BENCHMARK_TICKERS.items()
    ]
