from optimization import calculate_efficient_frontier, max_sharpe_ratio, optimize_portfolio
from data import load_data_from_file, fetch_data_from_api, fetch_risk_free_rate, calculate_returns, calculate_covariance_matrix, validate_data, get_tickers_list, handle_missing_data
from visualization import plot_efficient_frontier
"""
The main file of this package. It uses various functions accross all the files to accomplish different tasks easily.

Functions:
- efficient_frontier:
    Computes the efficient frontier for a given set of data. 
- load_data:
    Loads the data from the internet, checks it replaces any missing information.
- get_risk_free_rate:
    Fetch the risk-free rate from the internet
"""

def efficient_frontier(data, precision=50, show_graph=True, short_selling_allowed=False, risk_free_rate=None, calculate_max_sharpe_ratio=True):
    """
    Calculates the efficient frontier using a given DataFrame.
    This function can also calculates the maximum sharpe ratio, plot the efficient frontier and show the Capital Market Line.

    Paramaters:
    - data: pd.DataFrame
        The data containing all the periodic prices of the assets we want to invest in.
    - precision: int, optional, default=50
        The number of different portfolio we are going to calculate in order to compute the efficient frontier.
    - show_graph: bool, optional, default=False
        Indicates if the user wants to plant the graph showing the efficient frontier (and the Capital Market Line if max_sharpe_ratio is enabled).
    - short_selling_allowed: bool, optional, default=False:
        Indicates if the user wants to enable short selling (allowing the weights to be negative).
    - risk_free_rate: float, optional
        The average of the risk-free bond returns. Mandatory if the user wants to calculate the maximum sharpe ratio.
    - calculate_max_sharpe_ratio: bool, optional, default=True
        Indicates if the user want to search for the maximum sharpe ratio inside the efficient frontier. 
        Enables the Capital Market Line on the graph. 
    
    Returns:
    - frontier std: list
        The list of the standard variation of every portfolio calculated for the efficient frontier.
    - frontier weights: list
        The list of the optimized weights of every portfolio calculated for the efficient frontier.
    - optimized weights: list
        The weights corresponding to the portfolio having the maximum sharpe ratio.
    - optimized standard variation: float
        The standard variation of the portfolio having the maximum sharpe ratio. 
    - maximum sharpe ratio: float
        The maximum sharpe ratio of the efficient frontier.
    """
    if max_sharpe_ratio and risk_free_rate is None:
        raise ValueError("If you want to compute the maximum Sharpe ratio, provide the risk free rate")
    if not validate_data(data):
        raise ValueError("The data is invalid")
    returns = calculate_returns(data)
    cov_matrix = calculate_covariance_matrix(returns)
    expected_returns = returns.mean().values
    frontier_results = calculate_efficient_frontier(cov_matrix, expected_returns, num_points=precision, short_selling=short_selling_allowed)
    max_sharpe_results = {"opt_weights": None, "max_sharpe": None}
    if calculate_max_sharpe_ratio:
        max_sharpe_results = max_sharpe_ratio(cov_matrix, expected_returns, risk_free_rate=risk_free_rate, num_points=precision, short_selling=short_selling_allowed)
        max_sharpe = (max_sharpe_results["opt_std"], max_sharpe_results["opt_return"])
    if show_graph:
        plot_efficient_frontier(frontier_results['frontier'], max_sharpe=max_sharpe, risk_free_rate=risk_free_rate)
    return { "frontier std": frontier_results['frontier'], "frontier weights": frontier_results['weights list'], "optimized weights": max_sharpe_results['opt_weights'], "optimized standard variation": max_sharpe_results["opt_sdt"], "maximum sharpe ratio": max_sharpe_results['max_sharpe'] }

def load_data(tickers="USA", start_date="2022-01-01", end_date="2023-12-01", interval="1d"):
    """
    Fetch the data from Yahoo Finance using the yfinance API.

    Paramaters:
    - tickers: list or str, optional, default="USA"
        The list of all the tickers code of the assets we want to invest in, or the name of the country we want to invest in. 
    - start_date: str, optional, default="2022-01-01"
        The date at wich we start fetching the data.
    - end_date: str, optional, default="2023-12-01"
        The date at wich we stop fetching the data.
    - interval: str, optional, default="1d"
        The interval at wich we fetch the data.
    
    Returns:
    - data: pd.DataFrame
        A DataFrame containing the periodic prices of all the assets we want to invest in.
    """
    if not isinstance(tickers, list):
        tickers = get_tickers_list(tickers)['tickers']
    data = fetch_data_from_api(tickers=tickers, start_date=start_date, end_date=end_date, interval=interval)
    if not validate_data(data, tickers):
        raise ValueError("An error occured while fetching the data")
    data = handle_missing_data(data)
    return data

def get_risk_free_rate(rff="USA", interval="1d"):
    """
    Fetch the risk-free bond historic prices from the internet using the yfinance API and calculates the average of its returns.

    Paramaters:
    - rff: str, optional, default="USA"
        The ticker code of the bond or the name of the country we are going to get the risk-free bond from.
    - interval: str, optional, default="1d"
        The interval at wich we fetch the data.
    
    Returns:
    - risk_free_rate: float
        The average of the risk-free bond returns.
    """
    bond = get_tickers_list(rff)['risk free rate']
    risk_free_rate = fetch_risk_free_rate(bond, interval)
    return risk_free_rate

print(f"Weights: \n{efficient_frontier(load_data("South Korea", "2011-01-01", "2020-12-01", "1mo"), show_graph=False, short_selling_allowed=False, risk_free_rate=get_risk_free_rate("South Korea", "1mo"), calculate_max_sharpe_ratio=False)['optimized weights']}")
