import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.data_loader import fetch_stock_data
from src.portfolio import(calculate_daily_returns,equal_weight_portfolio,custom_weight_portfolio,buy_and_hold_portfolio,monthly_rebalanced_portfolio)
from src.metrics import (cumulative_returns, annualized_return, annualized_volatility,sharpe_ratio,max_drawdown,drawdown_series,rolling_volatility,portfolio_risk_decomposition,compute_risk_parity_weights)

if __name__=="__main__":

    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]

    start_date = "2020-01-01"
    end_date = "2024-01-01"

    prices= fetch_stock_data(tickers, start_date, end_date)

    # Fetch NIFTY index for market-based regime detection
    nifty_prices = fetch_stock_data(["^NSEI"], start_date, end_date)
    nifty_returns = calculate_daily_returns(nifty_prices).squeeze()  # Convert to Series

    returns = calculate_daily_returns(prices)

    equal_portfolio = equal_weight_portfolio(returns)
    equal_monthly = monthly_rebalanced_portfolio(returns, [0.25, 0.25, 0.25, 0.25])
    equal_weights = np.ones(len(tickers)) / len(tickers)

    custom_weights = [0.4,0.3,0.2,0.1]
    custom_portfolio = custom_weight_portfolio(returns,custom_weights)

    equal_buy_hold = buy_and_hold_portfolio(returns, [0.25, 0.25, 0.25, 0.25])

    equal_cumulative = cumulative_returns(equal_portfolio)
    equal_drawdown = drawdown_series(equal_cumulative)
    rolling_vol_equal = rolling_volatility(equal_portfolio)

    #ml regime detection using kmeans (based on NIFTY market conditions)
    nifty_rolling_vol = rolling_volatility(nifty_returns)
    nifty_rolling_ret = nifty_returns.rolling(window=30).mean()

    #combine features into dataframe
    features = pd.DataFrame({
        "RollingVol": nifty_rolling_vol,
        "RollingReturn": nifty_rolling_ret
    }).dropna()

    # Standardize features so both contribute equally to clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features[["RollingVol", "RollingReturn"]])

    kmeans=KMeans(n_clusters=3,random_state=42)
    kmeans.fit(scaled_features)
    features["Regime"]=kmeans.labels_
    features["Regime"] = features["Regime"].shift(1)

    # Identify clusters for 3 regimes: Crash, Sideways, Bull
    cluster_stats = features.groupby("Regime")[["RollingVol", "RollingReturn"]].mean()
    
    # Crash = highest volatility
    crash_cluster = cluster_stats["RollingVol"].idxmax()
    # Bull = highest return (among remaining)
    remaining = cluster_stats.drop(crash_cluster)
    bull_cluster = remaining["RollingReturn"].idxmax()
    # Sideways = the remaining one
    sideways_cluster = remaining.drop(bull_cluster).index[0]
    
    # Map numeric labels to descriptive names
    regime_labels = {
        crash_cluster: "Crash",
        bull_cluster: "Bull",
        sideways_cluster: "Sideways"
    }
    features["RegimeLabel"] = features["Regime"].map(regime_labels)

    # See what each cluster represents
    print("\n--- KMeans Regime Detection (3 Clusters) ---")
    regime_summary = cluster_stats.copy()
    regime_summary.index = regime_summary.index.map(regime_labels)
    print(regime_summary.sort_values("RollingVol", ascending=False))

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

    # Dynamic regime based portfolio with transaction costs
    # Crash → Risk Parity (defensive)
    # Sideways → 50% RP + 50% Equal Weight (balanced, slightly defensive)
    # Bull → Equal Weight (aggressive)
    
    blended_weights = 0.5 * np.array(rp_weights) + 0.5 * np.array(equal_weights)
    
    prev_weights = np.array(equal_weights)
    total_turnover = 0
    transaction_cost_rate = 0.001  # 0.1% per unit turnover
    min_hold_days = 20
    days_in_regime = 0
    current_regime = None
    actual_switches = 0  # Count actual switches after holding period

    dynamic_returns = []
    for date in returns.index:
        # Detect regime from ML model
        if date in features.index and pd.notna(features.loc[date, "RegimeLabel"]):
            detected_regime = features.loc[date, "RegimeLabel"]
        else:
            detected_regime = "Bull"  # Default for early period
        
        # Apply minimum holding period rule
        if current_regime is None:
            current_regime = detected_regime
            days_in_regime = 0
        elif detected_regime != current_regime and days_in_regime >= min_hold_days:
            current_regime = detected_regime
            days_in_regime = 0
            actual_switches += 1
        
        days_in_regime += 1
        
        # Select weights based on CURRENT regime (not detected)
        if current_regime == "Crash":
            current_weights = np.array(rp_weights)
        elif current_regime == "Sideways":
            current_weights = blended_weights
        else:  # Bull
            current_weights = np.array(equal_weights)
        
        # Calculate turnover
        turnover = np.sum(np.abs(current_weights - prev_weights))
        total_turnover += turnover
        
        # Portfolio return
        daily_return = returns.loc[date].dot(current_weights)
        
        # Subtract transaction cost
        transaction_cost = transaction_cost_rate * turnover
        daily_return -= transaction_cost
        
        dynamic_returns.append(daily_return)
        prev_weights = current_weights

    dynamic_portfolio = pd.Series(dynamic_returns, index=returns.index)
    dynamic_cumulative = cumulative_returns(dynamic_portfolio)

    # Count raw regime switches (before holding period filter)
    regime_series = features["RegimeLabel"].reindex(returns.index)
    raw_regime_switches = (regime_series != regime_series.shift(1)).sum()

    print("\nDynamic Strategy Performance (After Transaction Costs)")
    print("Annual Return:", annualized_return(dynamic_portfolio))
    print("Volatility:", annualized_volatility(dynamic_portfolio))
    print("Sharpe Ratio:", sharpe_ratio(dynamic_portfolio))
    print("Total Turnover:", round(total_turnover, 4))
    print("Raw Regime Switches:", raw_regime_switches)
    print("Actual Switches (after 20-day hold):", actual_switches)


    #Plots
    plt.figure(figsize=(12,16))
    plt.plot(equal_cumulative, label="Equal Weight Portfolio")
    plt.plot(custom_cumulative, label="Custom Weight Portfolio")
    plt.plot(dynamic_cumulative, label="ML Regime Strategy")
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
 