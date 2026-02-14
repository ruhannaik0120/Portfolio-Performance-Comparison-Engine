import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers,start_date,end_date):
    """
    Fetch historical stock price data for given tickers.
    
    Parameters:
        tickers (list): List of stock ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD'
        end_date (str): End date in 'YYYY-MM-DD'
    
    Returns:
        DataFrame: Adjusted close price data
    """   

    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    #Extract Adjusted Close prices
    adj_close = data["Close"]
    
    # Drop rows with missing values
    adj_close = adj_close.dropna()

    return adj_close
