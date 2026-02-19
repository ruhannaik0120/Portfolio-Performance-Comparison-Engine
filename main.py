"""
Portfolio Comparison Engine -- main entry point.
Runs 4 strategies, prints a clean comparison table, and shows 2 charts.
Enable RUN_RESEARCH_MODE in config.py for regime-switching analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    DEFAULT_TICKERS, START_DATE, END_DATE,
    TARGET_VOL, LOOKBACK_WINDOW, TRANSACTION_COST,
    MAX_LEVERAGE, RUN_RESEARCH_MODE,
)
from src.data_loader import fetch_stock_data
from src.portfolio import calculate_daily_returns
from src.metrics import (
    rolling_volatility,
    portfolio_risk_decomposition,
    compute_risk_parity_weights,
)
from src.strategies import (
    equal_weight_strategy,
    static_risk_parity_strategy,
    vol_targeted_risk_parity_strategy,
    walk_forward_risk_parity_strategy,
    regime_switching_strategy,
)
from src.factors import market_factor_analysis

# ------------------------------------------------------------------
# Research mode helpers
# ------------------------------------------------------------------

def run_research_analysis(returns, nifty_returns, rp_weights):
    """Extended analysis: regime switching, risk decomposition, rolling vol."""
    # Regime switching
    dyn_port, dyn_cum, dyn_dd, dyn_m = regime_switching_strategy(
        returns, nifty_returns, rp_weights, TRANSACTION_COST,
    )

    print("\n============== Research Mode ==============")

    # Regime cluster summary
    cluster_stats = dyn_m["cluster_stats"]
    regime_labels = dyn_m["regime_labels"]
    regime_summary = cluster_stats.copy()
    regime_summary.index = regime_summary.index.map(regime_labels)
    print("\nKMeans Regime Clusters:")
    print(regime_summary.sort_values("RollingVol", ascending=False))

    # Dynamic strategy metrics
    print("\nRegime Switching Strategy:")
    print(f"  Return:        {dyn_m['Return']:.4f}")
    print(f"  Volatility:    {dyn_m['Volatility']:.4f}")
    print(f"  Sharpe:        {dyn_m['Sharpe']:.4f}")
    print(f"  Max Drawdown:  {dyn_m['Max Drawdown']:.4f}")
    print(f"  Total Turnover:          {dyn_m['Total Turnover']}")
    print(f"  Raw Regime Switches:     {dyn_m['Raw Regime Switches']}")
    print(f"  Actual Switches (20-day): {dyn_m['Actual Switches (20-day hold)']}")

    # Risk decomposition (static RP)
    rp_risk_df = portfolio_risk_decomposition(returns, rp_weights)
    print("\nRisk Parity -- Risk Contributions:")
    print(rp_risk_df[["Weight", "Percent Contribution"]])

    # Rolling volatility chart
    eq_port = returns.dot([1 / returns.shape[1]] * returns.shape[1])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling_volatility(eq_port), label="Equal Weight")
    ax.plot(rolling_volatility(dyn_port), label="Regime Switching")
    ax.set_title("30-Day Rolling Volatility -- Research Strategies")
    ax.set_ylabel("Annualized Volatility")
    ax.legend()
    ax.grid(True)

    print("\n===========================================")
    plt.show()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":

    # 1. Data
    prices = fetch_stock_data(DEFAULT_TICKERS, START_DATE, END_DATE)
    returns = calculate_daily_returns(prices)

    # 2. Run strategies
    eq_port, eq_cum, eq_dd, eq_m, eq_w = equal_weight_strategy(returns)
    rp_port, rp_cum, rp_dd, rp_m, rp_w = static_risk_parity_strategy(returns)
    vt_port, vt_cum, vt_dd, vt_m, vt_w = vol_targeted_risk_parity_strategy(returns, TARGET_VOL)
    wf_port, wf_cum, wf_dd, wf_m, wf_w = walk_forward_risk_parity_strategy(
        returns, LOOKBACK_WINDOW, TARGET_VOL, TRANSACTION_COST, MAX_LEVERAGE,
    )

    # 3. Comparison table
    strategies = {
        "Equal Weight":       eq_m,
        "Static Risk Parity": rp_m,
        "Vol-Targeted RP":    vt_m,
        "Walk-Forward RP (Integrated)": wf_m,
    }
    cols = ["Return", "Volatility", "Sharpe", "Max Drawdown"]
    comparison = pd.DataFrame(
        {name: {c: m[c] for c in cols} for name, m in strategies.items()}
    ).T

    print("\n================ Portfolio Comparison ================\n")
    print(comparison.to_string(float_format="{:.4f}".format))
    print(f"\nWalk-Forward RP (Integrated) Total Turnover: {wf_m['Total Turnover']:.4f}")
    print("\n======================================================")

    # Factor Analysis (CAPM)
    nifty_prices = fetch_stock_data(["^NSEI"], START_DATE, END_DATE)
    nifty_returns = calculate_daily_returns(nifty_prices).squeeze()

    print("\nFactor Analysis (vs NIFTY):")
    factor_rows = {}
    factor_strategies = {
        "Equal Weight": eq_port,
        "Static RP": rp_port,
        "Vol-Targeted RP": vt_port,
        "Walk-Forward RP (Integrated)": wf_port
    }
    for name, strategy_returns in factor_strategies.items():
        factor_rows[name] = market_factor_analysis(strategy_returns, nifty_returns)

    factor_df = pd.DataFrame(factor_rows).T
    factor_df.columns = ["Alpha (Ann.)", "Beta", "R-squared"]
    print(factor_df.to_string(float_format="{:.4f}".format))
    print()

    # 4. Product charts (2 only)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot 1 -- Growth
    for label, cum in [("Equal Weight", eq_cum), ("Static RP", rp_cum),
                       ("Vol-Targeted RP", vt_cum), ("Walk-Forward RP (Integrated)", wf_cum)]:
        axes[0].plot(cum, label=label)
    axes[0].set_title(f"Portfolio Growth Comparison ({START_DATE[:4]}-{END_DATE[:4]})")
    axes[0].set_ylabel("Growth of 1 unit")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2 -- Drawdown
    for label, dd in [("Equal Weight", eq_dd), ("Static RP", rp_dd),
                      ("Vol-Targeted RP", vt_dd), ("Walk-Forward RP (Integrated)", wf_dd)]:
        axes[1].plot(dd, label=label)
    axes[1].set_title("Drawdown Comparison")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # 5. Research mode (optional)
    if RUN_RESEARCH_MODE:
        run_research_analysis(returns, nifty_returns, rp_w["Weight"].values)

    
 