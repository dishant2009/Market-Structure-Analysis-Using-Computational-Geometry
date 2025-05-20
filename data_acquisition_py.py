"""
Data Acquisition Module

This module contains functions for fetching financial data from Yahoo Finance,
calculating returns, and preparing data for analysis.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def get_stock_data(tickers, start_date, end_date, interval='1d'):
    """
    Fetch stock data from Yahoo Finance
    
    Parameters:
    -----------
    tickers : list
        List of stock ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    interval : str, optional
        Data interval, default is '1d' (daily)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing adjusted close prices for all tickers
    """
    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval)
    
    # Get adjusted close prices
    if len(tickers) > 1:
        prices = data['Adj Close']
    else:
        prices = data['Adj Close'].to_frame(name=tickers[0])
    
    # Fill missing values using forward fill followed by backward fill
    # This handles missing trading days and non-trading days
    prices = prices.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Downloaded data shape: {prices.shape}")
    return prices


def calculate_returns(prices, method='log'):
    """
    Calculate returns from price data
    
    Parameters:
    -----------
    prices : pandas.DataFrame
        DataFrame containing price data
    method : str, optional
        'log' for log returns, 'pct' for percentage returns
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing returns
    """
    if method == 'log':
        # Log returns are better for theoretical analysis and statistical properties
        returns = np.log(prices / prices.shift(1)).dropna()
    else:  # percentage returns
        # Percentage returns are easier to interpret
        returns = (prices / prices.shift(1) - 1).dropna()
    
    return returns


def get_market_data(tickers, start_date, end_date, interval='1d', returns_method='log'):
    """
    Get market data including prices and returns
    
    Parameters:
    -----------
    tickers : list
        List of stock ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    interval : str, optional
        Data interval, default is '1d' (daily)
    returns_method : str, optional
        Method to calculate returns, default is 'log'
    
    Returns:
    --------
    tuple
        (prices DataFrame, returns DataFrame)
    """
    # Get price data
    prices = get_stock_data(tickers, start_date, end_date, interval)
    
    # Calculate returns
    returns = calculate_returns(prices, method=returns_method)
    
    return prices, returns
