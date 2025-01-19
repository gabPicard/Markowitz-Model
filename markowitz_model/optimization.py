import numpy as np
from cvxopt import matrix, solvers
"""
This file contains functions to optimize a portfolio , to compute the efficient frontier, the calculate a sharpe ratio
and to find the maximum sharpe ratio of a given set of assets.

Functions:
- optimize_portfolio
    Finds the weights for the minimum variance point (with or without a specific return and with or without short selling).
- calculate_efficient_frontier
    Calculates every standard variation and weights possible for a set of assets.
- sharpe_ratio
    Calculates the sharpe ratio of a portfolio.
- max_sharpe_ratio
    Finds the maximum sharpe ratio possible for a set of asset.
- compute_return
    Computes the return of a portfolio given its weights and the expected returns of the assets.
- validate_input
    Runs some verifications on a input to avoid repetitions.
"""

def optimize_portfolio(cov_matrix, expected_returns=None, target_return=None, short_selling=False):
    """
    Computes the standard deviation and the weights for the minium variance point with the option to add a specific return and to allow short selling.
    Optimizes the portfolio using the Markowitz Model.

    Parameters: 
    - cov_matrix: np.ndarray
        A matrix containing the variance and covariances of every asset
    - expected_returns: np.ndarray, optional
        An array containing the expected return of every asset. Needed to compute a specific return. Set by default to None
    - target_return: float, optional, default=None
        A decimal number wich specifies the return we are optimizing the portfolio for. 
    - short_selling: bool, optional, default=False
        A boolean describing if short selling (weights can be negative) is allowed or no. 

    Returns:
    - std: float
        A number describing the standard deviation of the optimized portfolio
    - weights: np.ndarray
        An array containing the optimized weight for every asset
    """
    if validate_input(cov_matrix, "matrix"):
        num_assets = cov_matrix.shape[0]

        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvalues < 0):
            raise ValueError("Covariance matrix is not positive semi-definite.")

        if target_return is not None:
            min_return = np.min(expected_returns)
            max_return = np.max(expected_returns)
            if target_return < min_return or target_return > max_return:
                print(f"Target return {target_return} is infeasible.")
                return None

        P = matrix(cov_matrix)
        q = matrix(np.zeros(num_assets))
        A_list = [np.ones(num_assets)]
        b_list = [1.0]

        if target_return is not None and expected_returns is not None:
            A_list.append(expected_returns)
            b_list.append(target_return)

        A = matrix(np.vstack(A_list))
        b = matrix(b_list)

        G, h = None, None
        if not short_selling:
            G = matrix(-np.identity(num_assets))
            h = matrix(np.zeros(num_assets))

        try:
            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G, h, A, b)

            if solution['status'] != 'optimal':
                raise ValueError("Infeasible solution.")

            weights = np.array(solution['x']).flatten()
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            return {"std": std, "weights": weights}

        except Exception as e:
            print(f"Error in optimization: {e}")
            return None

    else:
        raise ValueError("Incorrect covariance matrix.")

def calculate_efficient_frontier (cov_matrix, expected_returns, num_points = 50, short_selling = False):
    """
    Computes every single combination of weights for every asset and their respective standard deviation.
    This function calls for a num_points number of feasible target returns the optimize_portfolio function.

    Parameters:
    - cov_matrix: np.ndarray
        A matrix containing the variance and covariances of every asset
    - expected_returns: np.ndarray
        An array containing the expected return of every asset
    - num_point: int, optional, default=50
        An integer describing the number of different portfolio that are going to be optimized. 
    - short_selling: bool, optional, default=False
        A boolean that describe if short selling (weights can be negative) is allowed. 
    
    Returns:
    - frontier: list
        A list containing the standard deviation of every portfolio optimized during the call of this function
    - weights list: list
        A list containing the optimized weights of every portfolio
    """
    if validate_input(cov_matrix, "matrix") and validate_input(expected_returns, "array") and cov_matrix.shape[0] == len(expected_returns):
        min_return = min(expected_returns)
        max_return = max(expected_returns)
        target_returns = np.linspace(min_return, max_return, num_points)

        frontier = []
        weights_list = []

        for target_return in target_returns:
            try:
                results = optimize_portfolio(cov_matrix, expected_returns, target_return, short_selling)
                frontier.append((results['std'], target_return))
                weights_list.append(results['weights'])
            except ValueError as e:
                print(f"Skipping target return {target_return}: {e}")

        return { "frontier": frontier, "weights list": weights_list }

def sharpe_ratio (std, expected_returns, weights, risk_free_rate):
    """
    Calculates the sharpe ratio of a given portfolio using its standard deviation, return and the risk-free rate.
    The sharpe ratio of the portfolio is a number describing how "well" the portfolio performs. 
    It is calculated by dividing the subtraction of the portfolio return by the risk-free rate by the standard deviation.

    Parameters:
    - std: float
        A number describing the standard deviation of the portfolio
    - expected_returns: np.ndarray
        An array containing the  expected return of every asset
    - weights: list
        A list containing the optimized weights of every asset of the portfolio
    - risk_free_rate: float
        A number describing the risk-free rate of the market
    
    Returns:
    - sharpe_ratio: float
        A number describing the sharpe ratio of the portfolio.
    """
    if validate_input(std, "int") and validate_input(expected_returns, "array") and validate_input(weights, "array") and validate_input(risk_free_rate, "int"): 
        portfolio_returns = np.dot(weights, expected_returns)
        sharpe_ratio = (portfolio_returns-risk_free_rate)/std
        

        return sharpe_ratio

def max_sharpe_ratio(cov_matrix, expected_returns, risk_free_rate, num_points=50, short_selling=False):
    """
    Finds the maximum sharpe ratio for a list of assets. 
    Calls the calculate_efficient_frontier function to get a list of every standard deviation and weights possible,
    then calls the sharpe_ratio function for every one of them, then finds the maximum ratio.

    Parameters:
    - cov_matrix: np.ndarray
        A matrix containing the variance and covariances of every asset.
    - expected_returns: np.ndarray
        An array containing the expected return of every asset.
    - risk_free_rate: float
        A number describing the risk-free rate of the market.
    - num_points: int, optional, default=50
        An integer describing the number of different portfolio that are going to be optimized.
    - short_selling: bool, optional, default=False
        A bollean describing if short selling (weights can be negative) is allowed. 
    
    Returns:
    - max_sharpe: float
        A number describing the maximum sharpe ratio found for these assets.
    - opt_return: float
        The return of the portfolio having the maximum sharpe ratio out of all of them.
    - opt_weights: list
        The list containing the weights of the portfolio having the maximum sharpe ratio.
    - opt_sdt: float
        The standard deviation of the portfolio having the maximum sharpe ratio.
    """
    if validate_input(cov_matrix, "matrix") and validate_input(expected_returns, "array") and validate_input(risk_free_rate, "int") and cov_matrix.shape[0] == len(expected_returns):
        max_index = None
        max_sharpe_ratio = None
        optimized_return = None
        optimized_std = None
        optimized_weights = None
            
        try:
            results = calculate_efficient_frontier(cov_matrix, expected_returns, num_points, short_selling)

            frontier = results['frontier']
            weights_list = results['weights list']

            sharpe_ratio_list = []
            for weights, std in zip(weights_list,frontier):
                sharpe_ratio_list.append(sharpe_ratio(std[0], expected_returns, weights, risk_free_rate))
                
            max_index = sharpe_ratio_list.index(max(sharpe_ratio_list))
            max_sharpe_ratio = max(sharpe_ratio_list)
            optimized_return = frontier[max_index][-1]
            optimized_std = frontier[max_index][0]
            optimized_weights = weights_list[max_index]

        except Exception as e:
            print(f"Error in `optimize_portfolio`: {str(e)}")

        return {"max_sharpe": max_sharpe_ratio, "opt_return": optimized_return, "opt_weights": optimized_weights, "opt_std": optimized_std}

def compute_return(weights, expected_returns):
    """
    Computes the return of a portfolio given its weights and the expected returns of the assets.

    Parameters:
    - weights: np.array
        The weights of the assets in the portfolio.
    - expected_returns: np.array
        The expected returns of the assets in the portfolio.

    Returns:
    - return: float
        The return of the portfolio.
    """
    return np.dot(weights, expected_returns)

def validate_input(input, type):
    """
    Allows to do quick verifications about an input given its type.
    Verifies a certain number of information about an input to avoid being repetitive in other functions.

    Parameters:
    - input: any
        The input we want to do verifications on.
    - type: str
        A string describing the expected type of the input. Changes what verifications are being made.
    
    Returns:
    -  validation: bool
        A boolean that describes if the input can be accepted or not.
    """
    validation = False
    if type == "matrix":
        if isinstance (input, np.ndarray) and input is not None and len(input) > 0 and input.shape[0] == input.shape[1]:
            validation = True
        else:
            raise ValueError("Invalid covariance matrix")
    elif type == "array":
        if isinstance (input, np.ndarray) and input is not None and len(input) > 0:
            validation = True
        else:
            raise ValueError("Invalid array")
    elif type == "int":
        if (isinstance (input, int) or isinstance (input, float)) and input is not None:
            validation = True
        else:
            raise ValueError("Invalid number")
    return validation

