import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import fetch_stock_data
from src.portfolio import(calculate_daily_returns,equal_weight_portfolio,custom_weight_portfolio,buy_and_hold_portfolio,monthly_rebalanced_portfolio)
from src.metrics import (cumulative_returns, annualized_return, annualized_volatility,sharpe_ratio,max_drawdown,drawdown_series,rolling_volatility,portfolio_risk_decomposition,compute_risk_parity_weights)

if __name__=="__main__":

    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]

    start_date = "2020-01-01"
    end_date = "2024-01-01"

    prices= fetch_stock_data(tickers, start_date, end_date)

    returns = calculate_daily_returns(prices)

    equal_portfolio = equal_weight_portfolio(returns)
    equal_monthly = monthly_rebalanced_portfolio(returns, [0.25, 0.25, 0.25, 0.25])

    custom_weights = [0.4,0.3,0.2,0.1]
    custom_portfolio = custom_weight_portfolio(returns,custom_weights)

    equal_buy_hold = buy_and_hold_portfolio(returns, [0.25, 0.25, 0.25, 0.25])

    equal_cumulative = cumulative_returns(equal_portfolio)
    equal_drawdown = drawdown_series(equal_cumulative)
    rolling_vol_equal = rolling_volatility(equal_portfolio)

    custom_cumulative = cumulative_returns(custom_portfolio)
    custom_drawdown = drawdown_series(custom_cumulative)
    rolling_vol_custom = rolling_volatility(custom_portfolio)

    risk_df = portfolio_risk_decomposition(returns, custom_weights)

    print("Equal Weight Portfolio:")
    print(equal_portfolio.head())
    
    print("\nCustom Weight Portfolio:")
    print(custom_portfolio.head())

    print("\nEqual Portfolio Growth:")
    print(equal_cumulative.head())

    print("\nCustom Portfolio Growth:")
    print(custom_cumulative.head())

    print("\nPerformance Comparison:")
    comparison_df = pd.DataFrame({
        "Equal Weight": [
            annualized_return(equal_portfolio),
            annualized_volatility(equal_portfolio),
            sharpe_ratio(equal_portfolio),
            max_drawdown(equal_cumulative)
        ],
        "Custom Weight": [
            annualized_return(custom_portfolio),
            annualized_volatility(custom_portfolio),
            sharpe_ratio(custom_portfolio),
            max_drawdown(custom_cumulative)
        ]
    }, index=["Annual Return", "Volatility", "Sharpe Ratio", "Max Drawdown"])
    print(comparison_df)
    
    # Risk Decomposition
    print("\nRisk Decomposition(Custom Portfolio):")
    print(risk_df)
    print("\nValidation Checks:")
    print("Sum of Total Contributions (Annual Vol):",
          risk_df["Total Contribution"].sum())
    print("Portfolio Annual Volatility (std):",
          annualized_volatility(custom_portfolio))
    print("Sum of Percent Contributions:",
          risk_df["Percent Contribution"].sum())
    

    print("\nRisk Parity Portfolio:")
    rp_weights = compute_risk_parity_weights(returns)
    # Display weights as DataFrame
    rp_weights_df = pd.DataFrame({
        "Risk Parity Weight": rp_weights
    }, index=returns.columns)
    print(rp_weights_df)
    rp_portfolio = custom_weight_portfolio(returns, rp_weights)
    rp_cumulative = cumulative_returns(rp_portfolio)
    
    rp_metrics_df = pd.DataFrame({
        "Risk Parity": [
            annualized_return(rp_portfolio),
            annualized_volatility(rp_portfolio),
            sharpe_ratio(rp_portfolio),
            max_drawdown(rp_cumulative)
        ]
    }, index=["Annual Return", "Volatility", "Sharpe Ratio", "Max Drawdown"])
    print(rp_metrics_df)

    # Validate risk parity check if risk contributions are equal
    rp_risk_df = portfolio_risk_decomposition(returns, rp_weights)
    print("\nRisk Parity Risk Contributions:")
    print(rp_risk_df[["Weight", "Percent Contribution"]])


    #Plots
    plt.figure(figsize=(12,16))
    plt.plot(equal_cumulative, label="Equal Weight Portfolio")
    plt.plot(custom_cumulative, label="Custom Weight Portfolio")
    plt.title("Portfolio Growth Comparison(2020-2024)")
    plt.xlabel("Date")
    plt.ylabel("Growth of ₹1 Investment")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(12,6))
    plt.plot(equal_cumulative, label="Equal Weight (Daily Rebalanced)")
    plt.plot(equal_buy_hold, label="Equal Weight(Buy & Hold)")
    plt.title("Rebalanced vs Buy & Hold Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(equal_drawdown, label="Equal Weight Drawdown")
    plt.plot(custom_drawdown, label="Custom Weight Drawdown")
    plt.title("Portfolio Drawdown Comparison (2020-2024)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(12,6))
    plt.plot(rolling_vol_equal, label="Equal Weight Rolling Volatility")
    plt.plot(rolling_vol_custom, label="Custom Weight Rolling Volatility")
    plt.title("30-Day Rolling Volatility Comparison")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(12, 6))
    plt.plot(equal_cumulative, label="Daily Rebalanced")
    plt.plot(equal_buy_hold, label="Buy & Hold")
    plt.plot(equal_monthly, label="Monthly Rebalanced")
    plt.title("Rebalancing Strategy Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10,6))
    plt.bar(risk_df.index, risk_df["Weight"],alpha=0.6,label="Weight")
    plt.bar(risk_df.index, risk_df["Percent Contribution"],alpha=0.6,label="Risk Contribution")
    plt.title("Weight vs Risk contribution(custom Portfolio)")
    plt.ylabel("Proportion")
    plt.legend()
    plt.grid(True)
    plt.show()








    plt.show()
