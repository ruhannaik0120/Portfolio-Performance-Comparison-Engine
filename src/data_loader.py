import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock price data for given tickers.

    Parameters:
        tickers (list): List of stock ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD'
        end_date (str): End date in 'YYYY-MM-DD'

    Returns:
        DataFrame: Adjusted close price data

    Raises:
        ValueError: If download returns no data or no valid columns remain.
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    except Exception as e:
        raise ConnectionError(f"Failed to download data from Yahoo Finance: {e}") from e

    if data is None or data.empty:
        raise ValueError(
            f"No price data returned for {tickers} between {start_date} and {end_date}. "
            "Check that the tickers and date range are valid."
        )

    # Extract Close prices
    if "Close" in data.columns or (hasattr(data.columns, "get_level_values")
                                    and "Close" in data.columns.get_level_values(0)):
        adj_close = data["Close"]
    else:
        # Single-ticker download may return flat columns
        adj_close = data

    # Ensure we always have a DataFrame (single ticker returns a Series)
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(name=tickers[0] if len(tickers) == 1 else "Close")

    # Drop rows with missing values
    adj_close = adj_close.dropna()

    if adj_close.empty:
        raise ValueError(
            f"All price data was NaN after cleaning for {tickers}. "
            "The tickers may be delisted or the date range too narrow."
        )

    return adj_close
