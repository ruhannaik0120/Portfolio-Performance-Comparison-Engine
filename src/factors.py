import numpy as np
import pandas as pd
import statsmodels.api as sm

def market_factor_analysis(portfolio_returns, market_returns):
    """
    Runs CAPM-style regression:
    Portfolio = alpha + beta * Market

    Returns:
        dict with alpha (annualized), beta, r_squared

    Raises:
        ValueError: If there is insufficient overlapping data for regression.
    """
    # Align dates
    df = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
    df.columns = ["Portfolio", "Market"]

    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} overlapping trading days between portfolio and market. "
            "Need at least 30 for a meaningful regression."
        )

    x = sm.add_constant(df["Market"])
    y = df["Portfolio"]

    try:
        model = sm.OLS(y, x).fit()
    except Exception as e:
        raise RuntimeError(f"CAPM regression failed: {e}") from e

    beta = model.params["Market"]
    alpha_daily = model.params["const"]
    alpha_annual = alpha_daily * 252
    r_squared = model.rsquared

    return {
        "Alpha(Annual)": alpha_annual,
        "Beta": beta,
        "R-squared": r_squared
    }