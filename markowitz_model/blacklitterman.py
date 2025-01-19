import numpy as np
import pandas as pd
import yfinance as yf
from data import load_data_from_file, calculate_returns, fetch_risk_free_rate, validate_data
"""
This file contains function to apply the Black Litterman Model, used to slightly modify the weights computed using a Markowitz Model 
according to the investor views regarding how the assets will perform.

Functions:
- compute_risk_aversion
    Computes the risk aversion of the investor using the historic data of the market.
- get_market_weights
    Get the weights of the assets in the market.
- compute_market_implied_return
    Computes the market-implied returns using the CAPM.
- apply_investor_view
    Modify the returns using the market-implied returns and the invesor views of the assets.
- black_litterman
    Use both of the previous functions to make the modification of the returns easier
"""

def compute_risk_aversion(market_ticker, start_date, end_date, period="1d", risk_free_rate="^IRX"):
    """
    Compute the risk aversion of the investor using the CAPM formula.

    Parameters:
    - market_ticker: str
        The ticker code of the market.
    - start_date: str
        The date at wich we start fetching the data.
    - end_date: str
        The date at wich we stop fetching the data.
    - period: str, optional, default="1d"
        The interval at wich we fetch the data.
    - risk_free_rate: str, optional, default="^IRX"
        The ticker code of the risk-free bond.
    
    Returns:
    - risk_aversion: float
        The risk aversion of the investor.
    """
    try:
        historic_data = load_data_from_file(market_ticker, start_date, end_date, period=period)
        if validate_data(historic_data):
            returns = calculate_returns(historic_data)
            expected_return = np.mean(returns, axis=0)
            variance = returns.var()
            risk_free_rate = fetch_risk_free_rate(risk_free_rate)
    except Exception as e:
        raise ValueError(f"An error occured while fetching the data: {e}")
    risk_aversion = (expected_return - risk_free_rate) / variance
    return risk_aversion

def get_market_weights(tickers):
    """
    Get the weights of the assets in the market.

    Parameters:
    - tickers: list
        The list of the tickers code of every asset we want to get informations about.
    
    Returns:
    - market_weights: np.array
        The weights of the assets in the market.
    """
    try:
        tickers_data = yf.Tickers(" ".join(tickers))
        market_caps = [tickers_data.tickers[t].info.get('marketCap', 0) for t in tickers]
        market_caps = np.nan_to_num(market_caps, nan=0)
        market_caps = np.array(market_caps, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"An error occured while fetching the data: {e}")
    return market_caps / np.sum(market_caps)

def compute_market_implied_return(weights, cov_matrix, risk_aversion=2.5):
    """
    Compute the market-implied returns using the CAPM formula.

    Parameters:
    - weights: np.array
        The weights of the assets in the portfolio.
    - cov_matrix: np.array
        The covariance matrix of the assets.
    - risk_aversion: float, optional, default=2.5

    Returns:
    - market_implied_returns: np.array
        The market-implied returns of the assets.
    """
    return risk_aversion * np.dot(cov_matrix, weights)

def apply_investor_view(market_returns, cov_matrix, p, q, omega):
    """
    Modify the returns using the market-implied returns and the invesor views of the assets.

    Parameters:
    - market_returns: np.array
        The market-implied returns of the assets.
    - cov_matrix: np.array
        The covariance matrix of the assets.
    - p: np.array
        The matrix of the views of the investor.
    - q: np.array
        The vector of the views of the investor.
    - omega: np.array
        The matrix of the uncertainty of the views of the investor.

    Returns:
    - adjusted_returns: np.array
        The modified returns of the assets.
    """
    adjusted_returns = np.linalg.inv(np.linalg.inv(cov_matrix)+np.dot(p.T,np.linalg.inv(omega).dot(p))).dot(np.dot(np.linalg.inv(cov_matrix),market_returns)+np.dot(p.T,np.linalg.inv(omega)).dot(q))
    return adjusted_returns

def black_litterman(tickers, risk_free_rate, cov_matrix, p, q, omega, market_ticker="^GSPC", start_date="2022-01-01", end_date="2023-12-01", period="1d"):
    """
    Use both of the previous functions to make the modification of the returns easier.

    Parameters:
    - tickers: list
        The list of the tickers code of every asset we want returns from.
    - risk_free_rate: float
        The risk free rate of the market.
    - cov_matrix: np.array 
        The covariance matrix of the assets.
    - p: np.array  
        The matrix of the views of the investor.
    - q: np.array
        The vector of the views of the investor.
    - omega: np.array
        The matrix of the uncertainty of the views of the investor.
    - confidence_level: float, optional, default=1.0
        The confidence level of the investor. Used to scale the uncertainty of the views.
    - market_ticker: str, optional, default="^GSPC"
        The ticker code of the market.
    - start_date: str, optional, default="2022-01-01"
        The date at wich we start fetching the data.
    - end_date: str, optional, default="2023-12-01"
        The date at wich we stop fetching the data.
    - period: str, optional, default="1d"
        The interval at wich we fetch the data.
    
    Returns:
    - adjusted_returns: np.array
        The modified returns of the assets.
    """
    risk_aversion = compute_risk_aversion(market_ticker, start_date, end_date, period=period, risk_free_rate=risk_free_rate)
    market_weights = get_market_weights(tickers)
    market_returns = compute_market_implied_return(market_weights, cov_matrix, risk_aversion=risk_aversion)
    adjusted_returns = apply_investor_view(market_returns, cov_matrix, p, q, omega)
    return adjusted_returns

