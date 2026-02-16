# ============================================================
# Portfolio Comparison Engine — Configuration
# ============================================================

DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "GOLDBEES.NS"]

START_DATE = "2020-01-01"
END_DATE = "2024-01-01"

# Strategy parameters
TARGET_VOL = 0.15
LOOKBACK_WINDOW = 252
TRANSACTION_COST = 0.001
MAX_LEVERAGE = 1.0

# Research mode toggle
RUN_RESEARCH_MODE = False

# ------------------------------------------------------------------
# Metric tooltips (for dashboard / UI display)
# ------------------------------------------------------------------
METRIC_TOOLTIPS = {
    # Portfolio comparison metrics
    "Return": "Annualized average return over the backtest period.",
    "Volatility": "Annualized standard deviation of daily returns — measures risk.",
    "Sharpe": "Return per unit of risk. Higher = better risk-adjusted performance.",
    "Max Drawdown": "Largest peak-to-trough loss. Shows worst-case downside.",
    # Factor analysis metrics
    "Alpha (Ann.)": "Excess return not explained by market exposure. Positive = strategy adds value beyond the market.",
    "Beta": "Sensitivity to NIFTY. Beta 1.0 = moves with market, lower = more defensive.",
    "R-squared": "% of strategy behavior explained by the market. Lower = more independent from NIFTY.",
    # Other
    "Total Turnover": "Cumulative weight changes across all rebalances. Higher = more trading cost.",
}
