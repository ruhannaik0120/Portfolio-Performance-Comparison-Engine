import pandas as pd
import numpy as np
from src.metrics import compute_risk_parity_weights

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

def walk_forward_rp_vol_targeted(
    returns,
    lookback=252,
    target_vol=0.15,
    vol_window=30,
    transaction_cost_rate=0.001,
    max_leverage=1.0
):
    """
    Walk-forward Risk Parity with:
    - Rolling covariance estimation (lookback window)
    - Monthly rebalancing
    - Daily volatility targeting
    - Transaction cost modeling

    Parameters:
        returns (DataFrame): asset returns
        lookback (int): rolling window for covariance estimation
        target_vol (float): annual volatility target
        vol_window (int): rolling window for realized portfolio vol
        transaction_cost_rate (float): cost per unit turnover
        max_leverage (float): maximum scaling factor (1.0 = no leverage)

    Returns:
        portfolio_returns (Series)
        total_turnover (float)

    Raises:
        ValueError: If data has fewer rows than lookback window.
    """
    if len(returns) < lookback:
        raise ValueError(
            f"Not enough data for walk-forward: {len(returns)} rows but lookback is {lookback}. "
            "Use a shorter lookback or a wider date range."
        )

    portfolio_returns = []
    total_turnover = 0

    prev_weights = None
    current_weights = None

    # Identify month-end rebalance dates
    rebalance_dates = returns.groupby(
        returns.index.to_period("M")
    ).tail(1).index

    for i in range(len(returns)):

        date = returns.index[i]

        # --- Rebalance monthly using past data only ---
        if date in rebalance_dates and i >= lookback:

            window_data = returns.iloc[i - lookback:i]

            new_weights = compute_risk_parity_weights(window_data)

            # Transaction cost (if not first allocation)
            if prev_weights is not None:
                turnover = np.sum(np.abs(new_weights - prev_weights))
                total_turnover += turnover
            else:
                turnover = 0

            current_weights = new_weights
            prev_weights = new_weights

        # Skip until first allocation is available
        if current_weights is None:
            portfolio_returns.append(0)
            continue

        # --- Compute daily portfolio return ---
        daily_return = returns.iloc[i].dot(current_weights)

        # --- Apply transaction cost on rebalance day ---
        if date in rebalance_dates and i >= lookback:
            daily_return -= transaction_cost_rate * turnover

        # --- Volatility targeting ---
        temp_series = pd.Series(portfolio_returns + [daily_return])
        rolling_vol = temp_series.rolling(vol_window).std() * np.sqrt(252)

        realized_vol = rolling_vol.iloc[-1]

        if not np.isnan(realized_vol) and realized_vol > 0:
            scale = target_vol / realized_vol
            scale = min(scale, max_leverage)   # Cap leverage
        else:
            scale = 1.0

        daily_return *= scale

        portfolio_returns.append(daily_return)

    portfolio_returns = pd.Series(
        portfolio_returns,
        index=returns.index
    )

    return portfolio_returns, total_turnover