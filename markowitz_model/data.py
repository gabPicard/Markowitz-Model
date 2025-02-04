import pandas as pd
import yfinance as yf
import numpy as np
"""
This file contains the functions needed to get the data (from a file or from Yahoo Finance), to validate it and to calculate
the different matrices that are going to be needed in other files.

Functions:
- load_data_from_file
    Loads the data from a given file.
- fetch_data_from_api
    Gets the data from Yahoo Finance using the yfinance API.
- fetch_risk_free_rate
    Gets the risk-free rate from Yahoo Finance.
- validate_data
    Runs some verification on the given data.
- calculate_return_column
    Transforms a column of prices into a column of periodic returns.
- calculate_returns
    Transforms all the price of a DataFrame into returns using the calculate_return_column function.
- calculate_covariance_matrix
    Calculates the covariance matrix.
- handle_missing_data
    Replace the missing data using various methods.
- save_data
    Saves the given data in a .csv file.
"""

def load_data_from_file(file_path, index_col=0, parse_dates=True):
    """
    Load data from a CSV file into a pandas DataFrame.
    This function reads a CSV file from the specified file path and loads it into a pandas DataFrame. 

    Parameters:
    - file_path: str
        The path to the CSV file to be loaded.
    - index_col: int or str, optional, default=0
        Column to use as the row labels of the DataFrame. By default, the 
        first column is used as the index.
    - parse_dates: bool, optional, default=True
        If True, attempt to parse dates in the index column or date fields.

    Returns:
    - pandas.DataFrame
        A DataFrame containing the data from the CSV file.
    """
    try:
        data = pd.read_csv(file_path, index_col=index_col, parse_dates=parse_dates)
        if data.empty:
            raise ValueError("the .csv file is empty")
        return data
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")

def fetch_data_from_api(tickers, start_date, end_date, interval="1d"):
    """
    Fetch the data from Yahoo Finance using the yfinance API

    Parameters:
    - tickers: list
        The list of the tickers code of every asset we want to get informations about.
    - start_date: str
        A string describing the date at wich we start fetching data.
    - end_date: str
        Indicates the data at wich we stop fetching data.
    - interval: str, optional, default="1d"
        Indicates at wich period we fetch the data.
    
    Returns:
    - data: pd.DataFrame
        A DataFrame contaning the periodic price of every asset specified in the tickers list.
    """
    if not tickers:
        tickers = ["^GSPC", "^IXIC"]
    try:
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval)
        if "Adj Close" in data.columns:
            data = data["Adj Close"]
        if data.empty:
            raise ValueError("No data found for the specified tickers and date range.")
        return data
    
    except Exception as e:
        raise ValueError(f"Error fetching data from Yahoo Finance: {e}")

def fetch_risk_free_rate(bond="^IRX", period="5y"):
    """
    Fetch the risk-free price and calculates its rate using the calculate_return_column function

    Parameters:
    - bond: str, optional, default="^IRX"
        The ticker code for the risk-free bond we are using.
    - period: str, optional, default="1d"
        The interval at wich we get the data.
    
    Returns:
    - risk_free_rate: float
        The average of the return of every historic price of the risk-free bond fetched.
    """
    treasury_data = yf.Ticker(bond)
    historical_data = treasury_data.history(period=period)
    risk_free_rate = historical_data["Close"].pct_change().mean()
    return risk_free_rate

def validate_data(data, required_columns=None):
    """
    Verifies if the data is a pd.DataFrame and if columns are missing.

    Paramaters:
    - data: pd.DataFrame
        The DataFrame we want to run verifications on.
    - required_columns: list, optional
        The list of every column required for the data to be validated.
    
    Returns:
    - True or an Error if a column is missing/the data isn't a DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be pandas DataFrame")
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
    if data.isnull().any().any():
        raise ValueError("Data contains missing values.")
    return True

def calculate_return_column(column):
    """
    Calculates the return using the prices inside a column. 
    The return at time T is equal to the log of the price at T divided by the price at T-1.

    Parameters:
    - column: pd.DataFrame
        A column containing the periodic prices we want to transform into returns.
    
    Returns:
    - a pd.Series of every periodic return.
    """
    try:
        returns = [np.nan]  
        for k in range(1, len(column)):
            if column.iloc[k] > 0 and column.iloc[k-1] > 0:  
                returns.append(np.log(column.iloc[k] / column.iloc[k-1]))
            else:
                returns.append(np.nan)
        return pd.Series(returns, index=column.index)
    except Exception as e:
        raise ValueError(f"Error calculating returns for column: {e}")

def calculate_returns(price_data):
    """
    Transforms a DataFrame filled with periodic prices into periodic returns.
    Using the calculate_return_column function for every column of the DataFrame.

    Parameters:
    - price_data: pd.DataFrame
        A DataFrame containing all the periodic prices we want to transform.
    
    Returns:
    - price_data: pd.DataFrame
        A DataFrame now containing every periodic return.
    """
    try:
        for column_name in price_data.columns:
            if column_name != "Date":
                price_data[column_name] = pd.to_numeric(price_data[column_name], errors="coerce")
        
        for column_name in price_data.columns:
            if column_name != "Date":
                price_data[column_name] = calculate_return_column(price_data[column_name])
        
        return price_data
    except Exception as e:
        raise ValueError(f"Error calculating returns for DataFrame: {e}")

def calculate_covariance_matrix(returns):
    """
    Calculates the covariance matrix of every asset using their periodic returns.

    Parameters:
    - returns: pd.DataFrame
        A DataFrame containing the periodic returns of every asset.
    
    Returns:
    - a np.ndarray containing all variances and covariances.
    """
    try:
        return returns.cov().values
    except Exception as e:
        raise ValueError(f"Error calculating the covariance matrix: {e}")
    
def handle_missing_data(data, method="bfill"):
    """
    Fills the missing data (NaN) using two methods: backward fill by default (uses the next data) and forward fill (uses the previous data).

    Parameters:
    - data: pd.DataFrame
        The DataFrame we want to ensure is filled.
    method: str, optional, default="bfill"
        The method we are using. Set to "bfill" by default (backward fill).
    
    Returns:
    - a pd.DataFramef filled.
    """
    if method == "ffill" or method == "bfill":
        return data.fillna(method)
    elif method == "drop":
        return data.dropna()
    else:
        raise ValueError("Invalid method. Please choose from 'ffill', 'bfill' or 'drop'")

def save_data(data, file_path="data.csv"):
    """
    Saves the given data into a .csv file.

    Parameters:
    - data: pd.DataFrame
        The data we want to save.
    - file_path: str, optional, default="data.csv"
        The name of the file where the data will be saved.
    """
    try:
        data.to_csv(file_path)
    except Exception as e:
        raise ValueError("Failed to save the data: {e}")