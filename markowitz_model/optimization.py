import numpy as np
from cvxopt import matrix, solvers

def optimize_portfolio (cov_matrix, expected_returns=None, target_return=None, short_selling=False):
    if validate_input(cov_matrix, "matrix"):
        num_asset = cov_matrix.shape[0]

        p = matrix(cov_matrix)
        q = matrix(np.zeros(num_asset))

        a_list = [np.ones(num_asset)]
        b_list = [1.0]

        if target_return is not None and expected_returns is not None:
            a_list.append(expected_returns)
            b_list.append(target_return)
        
        a = matrix(np.vstack(a_list))
        b = matrix(b_list)

        g = None
        h = None

        if not short_selling:
            g = matrix(-np.identity(num_asset))
            h = matrix(np.zeros(num_asset))

        std = 0
        weights = []
        
        try:
            solvers.option['show_progress'] = False
            solution = solvers.qp(p, q, g, h, a, b)

            if solution['status'] != 'optimal':
                raise ValueError("Infeasible solution")
            
            weights = np.array(solution['x']).flatten()
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        except Exception as e:
            print(f"Target return {target_return} is infeasible")
        
        return { "std": std, "weights": weights }

    else:
        raise ValueError("Incorrect covariance matrix")


def efficient_frontier (cov_matrix, expected_returns, num_points = 50, short_selling = False):
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
    if validate_input(std, "int") and validate_input(expected_returns, "array") and validate_input(weights, "array") and validate_input(risk_free_rate, "int"): 
        portfolio_returns = np.dot(weights, expected_returns)
        sharpe_ratio = (portfolio_returns-risk_free_rate)/std

        return sharpe_ratio

def max_sharpe_ratio(cov_matrix, expected_returns, risk_free_rate, num_points=50, short_selling=False):
    if validate_input(cov_matrix, "matrix") and validate_input(expected_returns, "array") and validate_input(risk_free_rate, "int") and cov_matrix.shape[0] == len(expected_returns):
        max_index = None
        max_sharpe_ratio = None
        optimized_return = None
        optimized_std = None
        optimized_weights = None
            
        try:
            results = efficient_frontier(cov_matrix, expected_returns, num_points, short_selling)

            frontier = results['frontier']
            weights_list = results['weights list']

            sharpe_ratio_list = []
            for weights, std in zip(weights_list,frontier):
                sharpe_ratio_list.append(sharpe_ratio(std[0], expected_returns, weights, risk_free_rate))
                
            max_index = sharpe_ratio_list.index(max(sharpe_ratio_list))
            max_sharpe_ratio = max(sharpe_ratio_list)
            optimized_return = frontier[max_index][-1]
            optimized_std = frontier[max_index][-1]
            optimized_weights = weights_list[max_index]

        except Exception as e:
            print(f"Error in `optimize_portfolio`: {str(e)}")

        return {"max_sharpe": max_sharpe_ratio, "opt_return": optimized_return, "opt_weights": optimized_weights, "opt_std": optimized_std}

def validate_input(input, type):
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
            