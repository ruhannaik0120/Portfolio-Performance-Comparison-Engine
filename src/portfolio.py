import pandas as pd
import numpy as np

def calculate_daily_returns(price_data):
    """
    Calculate daily percentage returns from price data.
    
    Parameters:
        price_data (DataFrame): Stock price data
    
    Returns:
        DataFrame: Daily returns
    """

    returns = price_data.pct_change()

    returns= returns.dropna()

    return returns

def equal_weight_portfolio(returns):
    """
    Construct equal-weight portfolio returns.
    
    Parameters:
        returns (DataFrame): Daily stock returns
    
    Returns:
        Series: Portfolio daily returns
    """
    
    num_assets = returns.shape[1]
    
    weights = [1 / num_assets] * num_assets
    
    portfolio_returns = returns.dot(weights)
    
    return portfolio_returns

def custom_weight_portfolio(returns, weights):
    """
    Construct custom-weight portfolio returns.
    
    Parameters:
        returns (DataFrame): Daily stock returns
        weights (list): Custom weights for each asset
    
    Returns:
        Series: Portfolio daily returns
    """

    if len(weights) != returns.shape[1]:
        raise ValueError("Number of weights must match number of assets.")
    
    if round(sum(weights), 5) != 1:
        raise ValueError("Weights must sum to 1.")
    
    portfolio_returns = returns.dot(weights)

    return portfolio_returns

def buy_and_hold_portfolio(returns,weights):
    """
    Simulate buy-and-hold portfolio (no rebalancing).
    
    Parameters:
        returns (DataFrame): Daily stock returns
        weights (list): Initial weights
    
    Returns:
        Series: Portfolio cumulative value over time
    """
    if len(weights) !=returns.shape[1]:
        raise ValueError("Weights must match number of assests")
    
    if round(sum(weights),5) !=1:
        raise ValueError("Weights must sum to 1")
    
    #Initial portfolio value
    portfolio_value = 1.0
    
    # Initial asset values
    asset_values = np.array(weights) * portfolio_value

    portfolio_values = []

    for daily_returns in returns.values:
        #update each asset value
        asset_values = asset_values * (1+daily_returns)

        #compute total portfolio value
        portfolio_value = asset_values.sum()

        portfolio_values.append(portfolio_value)

    return pd.Series(portfolio_values,index=returns.index)

def monthly_rebalanced_portfolio(returns, weights):
    """
    Simulate monthly rebalanced portfolio.
    
    Parameters:
        returns (DataFrame): Daily stock returns
        weights (list): Target weights
    
    Returns:
        Series: Portfolio cumulative value over time
    """
    if len(weights) != returns.shape[1]:
        raise ValueError("Weights must match number of assets.")
    
    if round(sum(weights), 5) != 1:
        raise ValueError("Weights must sum to 1.")
    
    portfolio_value = 1.0
    asset_values = np.array(weights) * portfolio_value

    portfolio_values = []

    # Identify month-end dates
    month_end = returns.groupby(returns.index.to_period('M')).tail(1).index
    
    for date, daily_returns in zip(returns.index, returns.values):
        # Update asset values
        asset_values = asset_values * (1 + daily_returns)

        portfolio_value = asset_values.sum()
        portfolio_values.append(portfolio_value)

        # If month-end, rebalance
        if date in month_end:
            asset_values = portfolio_value * np.array(weights)

    return pd.Series(portfolio_values, index=returns.index)

