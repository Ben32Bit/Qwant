# FX pairs — (yfinance ticker, invert)
# invert=True  → ticker is FOREIGN/USD (e.g. EURUSD=X = USD per EUR → 1/rate = EUR per USD)
# invert=False → ticker is USD/FOREIGN (e.g. USDSGD=X = SGD per USD → use directly)
FX_PAIRS = {
    "EUR": ("EURUSD=X", True),
    "GBP": ("GBPUSD=X", True),
    "JPY": ("USDJPY=X", False),
    "CHF": ("USDCHF=X", False),
    "CAD": ("USDCAD=X", False),
    "AUD": ("AUDUSD=X", True),
    "NZD": ("NZDUSD=X", False),
    "SGD": ("USDSGD=X", False),
    "HKD": ("USDHKD=X", False),
    "CNH": ("USDCNH=X", False),
}

BENCHMARK_TICKERS = {
    "SPY": "S&P 500 (SPY)",
    "QQQ": "Nasdaq 100 (QQQ)",
    "IWM": "Russell 2000 (IWM)",
    "AGG": "US Aggregate Bond (AGG)",
    "EFA": "International Developed (EFA)",
    "EEM": "Emerging Markets (EEM)",
    "GLD": "Gold (GLD)",
    "TLT": "20+ Year Treasury (TLT)",
    "VTI": "Total US Market (VTI)",
    "BND": "Total Bond Market (BND)",
}

REBALANCE_FREQUENCIES = ["daily", "weekly", "monthly", "quarterly", "annually", "none"]

STRATEGY_TYPES = ["custom", "min_variance", "max_sharpe", "risk_parity", "inverse_vol", "equal_weight"]

TRADING_DAYS_PER_YEAR = 252

DEFAULT_INITIAL_CAPITAL = 10_000
DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_BENCHMARK = "SPY"
DEFAULT_REBALANCE_FREQUENCY = "quarterly"
DEFAULT_STRATEGY = "custom"
