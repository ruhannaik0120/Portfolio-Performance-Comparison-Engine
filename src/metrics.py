import numpy as np
import pandas as pd
from scipy.optimize import minimize

def cumulative_returns(portfolio_returns):
    """
    Compute cumulative growth of portfolio.
    """
    return (1+portfolio_returns).cumprod()

def annualized_return(portfolio_returns,trading_days=252):
    avg_daily_return = portfolio_returns.mean()
    return avg_daily_return * trading_days

def annualized_volatility(portfolio_returns, trading_days=252):
    daily_vol = portfolio_returns.std()
    return daily_vol * np.sqrt(trading_days)

def sharpe_ratio(portfolio_returns, risk_free_rate=0.0, trading_days=252):
    ann_return = annualized_return(portfolio_returns, trading_days)
    ann_vol = annualized_volatility(portfolio_returns, trading_days)
    return (ann_return - risk_free_rate) / ann_vol

def max_drawdown(cumulative_returns):
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return drawdown.min()

def drawdown_series(cumulative_returns):
    """
    Compute drawdown series over time.
    """
    rolling_max=cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return drawdown

def rolling_volatility(portfolio_returns, window=30, trading_days=252):
    rolling_std = portfolio_returns.rolling(window=window).std()
    return rolling_std * np.sqrt(trading_days)

def portfolio_risk_decomposition(returns,weights):
    """
    Compute marginal and total risk contribution of each asset.
    
    Parameters:
        returns (DataFrame): Asset daily returns
        weights (list or array): Portfolio weights
    
    Returns:
        DataFrame: Risk contribution breakdown
    """
    #Convert weights to numpy array
    weights = np.array(weights)

    if len(weights) != returns.shape[1]:
        raise ValueError("Number of weights must match number of assets.")
    
    if round(weights.sum(),5) != 1:
        raise ValueError("Weights must sum to 1.")
    
    # Covariance matrix
    trading_days=252
    cov_matrix = returns.cov()*trading_days
    
    # Portfolio variance
    portfolio_variance = weights.T @ cov_matrix.values @ weights
    
    # Portfolio volatility
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Marginal Contribution to Risk
    marginal_contribution = (cov_matrix.values @ weights) / portfolio_volatility
    
    # Total Risk Contribution
    total_contribution = weights * marginal_contribution
    
    # Percentage Contribution
    percent_contribution = total_contribution / portfolio_volatility

    risk_df = pd.DataFrame({
        "Weight": weights,
        "Marginal Contribution": marginal_contribution,
        "Total Contribution": total_contribution,
        "Percent Contribution": percent_contribution
    }, index=returns.columns)
    
    return risk_df

def compute_risk_parity_weights(returns):
    """
    Compute long-only risk parity weights.
    """
    trading_days=252
    cov_matrix=returns.cov()*trading_days
    cov_matrix=cov_matrix.values
    n=cov_matrix.shape[0]

    #Initial guess: equal weights
    init_weights=np.ones(n)/n

    def objective(weights):
        portfolio_variance=weights.T @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)

        #Marginal contribution
        mctr=(cov_matrix @ weights)/ portfolio_vol
        trc= weights*mctr
        percent_contrib=trc/portfolio_vol

        target=np.ones(n)/n
        return np.sum((percent_contrib-target)** 2)
    
    #constraints
    constraints=({
        'type':'eq',
        'fun':lambda w: np.sum(w)-1
    })

    bounds = tuple((0, 1) for _ in range(n))

    result = minimize(objective,
                      init_weights,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

    return result.x