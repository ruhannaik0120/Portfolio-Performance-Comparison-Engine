import numpy as np
import pandas as pd
import statsmodels.api as sm

def market_factor_analysis(portfolio_returns,market_returns):
    """
    Runs CAPM-style regression:
    Portfolio = alpha + beta * Market

    Returns:
        dict with alpha (annualized), beta, r_squared
    """
    #Align dates
    df=pd.concat([portfolio_returns,market_returns],axis=1).dropna()
    df.columns=["Portfolio","Market"]
    x=sm.add_constant(df["Market"])
    y=df["Portfolio"]
    model=sm.OLS(y,x).fit()
    beta=model.params["Market"]
    alpha_daily=model.params["const"]
    alpha_annual = alpha_daily*252
    r_squared=model.rsquared

    return{
        "Alpha(Annual)":alpha_annual,
        "Beta":beta,
        "R-squared":r_squared
    }