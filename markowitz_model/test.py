import numpy as np
import pandas as pd
from optimization import optimize_portfolio, calculate_efficient_frontier, sharpe_ratio, compute_return
from data import fetch_data_from_api, load_data_from_file, fetch_risk_free_rate, calculate_returns, calculate_covariance_matrix, validate_data
from visualization import plot_efficient_frontier
from blacklitterman import black_litterman, compute_risk_aversion, get_market_weights, compute_market_implied_return, apply_investor_view

def test_data():
    """
    Test all main functions in the data.py file.
    """
    print("Testing data functions...")

    print("Testing fetch_data_from_api...")
    tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    api_data = fetch_data_from_api(tickers, start_date, end_date)
    assert not api_data.empty, "Data fetching failed: price data is empty."
    assert validate_data(api_data), "Data fetching failed: data is invalid."
    print(f"Data fetched successfully")
    
    print("Testing load_data_from_file...")
    file_path = "test_dataset.csv"
    file_data = load_data_from_file(file_path)
    assert not file_data.empty, "Data loading failed: price data is empty."
    print(f"Data loaded successfully")

    print("Testing fetch_risk_free_rate...")
    risk_free_rate = fetch_risk_free_rate()
    assert risk_free_rate is not None, "Risk-free rate fetching failed: rate is None."
    print(f"Risk-free rate fetched successfully: {risk_free_rate}")

    print("Testing calculate_returns and calculate_covariance_matrix...")
    cov = [[ 0.16018894, 0.16593505, 0.0133169, 0.00454255, 0.06487117, 0.05605724, 0.00309595, 
                0.1462586, 0.04187515, 0.04007777, 0.02898524, -0.09921903],
           [ 0.16593505, 0.35489862, -0.00255119, -0.0195673, 0.04454238, 0.03828978, 
                -0.05449617, 0.2074613, 0.06244346, 0.03929364, 0.01526935, -0.20217072],
           [ 0.0133169, -0.00255119, 0.07825146, 0.02850866, 0.03674105, 0.02592467,
                0.02181397, 0.0326025, -0.00178305, 0.06529363, 0.02231728, 0.01068286],
           [ 0.00454255, -0.0195673, 0.02850866, 0.05234446, 0.0166747, 0.01232114,
                0.01118819, -0.01976896, -0.00945556, 0.02143252, 0.01114044, 0.03000372],
           [ 0.06487117, 0.04454238, 0.03674105, 0.0166747, 0.07935798, 0.02823845,
                0.02333469, 0.04195514, 0.02568816, 0.03101479, 0.01916366, -0.00732771],
           [ 0.05605724, 0.03828978, 0.02592467, 0.01232114, 0.02823845, 0.12844677,
                0.00421495, 0.08454252, 0.02268077, 0.0026338, 0.02020915, -0.04425649],
           [ 0.00309595, -0.05449617, 0.02181397, 0.01118819, 0.02333469, 0.00421495,
                0.04869933, -0.02972432, 0.00873426, 0.01599086, 0.01423256, 0.04799236],
           [ 0.1462586, 0.2074613, 0.0326025, -0.01976896, 0.04195514, 0.08454252,
                -0.02972432, 0.22587363, 0.03620984, 0.05746867, 0.02765994, -0.15496742],
           [ 0.04187515, 0.06244346, -0.00178305, -0.00945556, 0.02568816, 0.02268077,
                0.00873426, 0.03620984, 0.04820918, -0.00457638, 0.01069336, -0.03073577],
           [ 0.04007777, 0.03929364, 0.06529363, 0.02143252, 0.03101479, 0.0026338,
                0.01599086, 0.05746867, -0.00457638, 0.08958569, 0.02248199, -0.01628837],
           [ 0.02898524, 0.01526935, 0.02231728, 0.01114044, 0.01916366, 0.02020915,
                0.01423256, 0.02765994, 0.01069336, 0.02248199, 0.01413139, -0.00577645],
           [-0.09921903, -0.20217072, 0.01068286, 0.03000372, -0.00732771, -0.04425649,
                0.04799236, -0.15496742, -0.03073577, -0.01628837, -0.00577645, 0.16740995 ]]
    cov = np.array(cov)
    return_matrix = calculate_returns(file_data)
    cov_matrix = calculate_covariance_matrix(return_matrix)
    assert np.allclose(cov, cov_matrix), "Covariance matrix calculation failed: matrices are not equal."
    print("Returns and covariance matrix calculated successfully")
    return cov_matrix, risk_free_rate

def test_optimization(cov_matrix, risk_free_rate):
    """
    Test all main functions in the optimization.py file.
    """
    print("Testing optimization functions...")
    print("Testing with short-selling allowed...")
    optimized_weights_short_selling = [-0.28248981, 0.14321358, -0.34015795, -0.10184402, 0.12198836, 0.01035752, -0.22306061, 0.01874789, -0.13192143, 0.01233304, 1.57579341, 0.19704003]
    optimized_weights_short_selling = np.array(optimized_weights_short_selling)
    weights = optimize_portfolio(cov_matrix, short_selling=True)['weights']
    assert np.allclose(optimized_weights_short_selling, weights, atol=1e-4), "Optimization failed: weights are not equal."
    print(f"Optimization successful: {weights}")
    print("Testing with short-selling not allowed and target return...")
    expected_returns = np.array([0.1, 0.12, 0.08, 0.07, 0.09, 0.13, 0.06, 0.11, 0.08, 0.1, 0.09, 0.04])
    target_return = 0.05
    optimized_weights_no_short_selling = [1.37071103e-08, 5.99595491e-06, 1.11044088e-08, 8.14292734e-07, 6.20903528e-09, 2.16671625e-09, 4.98724889e-01, 5.89738624e-08, 6.24790682e-04, 4.30143832e-09, 6.73527727e-09, 5.00643407e-01]
    optimization = optimize_portfolio(cov_matrix, expected_returns=expected_returns, target_return=target_return, short_selling=False)
    weights = optimization['weights']
    std = optimization['std']
    assert np.allclose(optimized_weights_no_short_selling, weights, atol=1e-4), "Optimization failed: weights are not equal."
    print(f"Optimization successful: {weights}")

    print("Testing calculate_efficient_frontier...")
    expected_frontier = [(0.12682329300314682, 0.04), (0.07470316738231529, 0.08499999999999999), (0.09588550294271031, 0.13)]
    frontier = calculate_efficient_frontier(cov_matrix, expected_returns=expected_returns, num_points = 3, short_selling=True)['frontier']
    assert np.allclose(expected_frontier, frontier, atol=1e-4), "Efficient frontier calculation failed: frontiers are not equal."
    print("Efficient frontier calculated successfully")

    print("Testing compute_return...")
    expected_return = compute_return(weights, expected_returns)
    assert np.isclose(expected_return, target_return, atol=1e-4), "Return calculation failed: returns are not equal."
    print(f"Return calculated successfully: {expected_return}")

    print("Testing sharpe_ratio...")
    expected_sharpe_ratio = (target_return - risk_free_rate) / std
    sharpe = sharpe_ratio(std, expected_returns, weights, risk_free_rate)
    assert np.isclose(expected_sharpe_ratio, sharpe, atol=1e-4), "Sharpe ratio calculation failed: ratios are not equal."
    print(f"Sharpe ratio calculated successfully: {sharpe}")

def test_blacklitterman():
    """
    Test all functions in the blacklitterman.py file.
    """
    print("Testing compute_risk_aversion...")
    market_ticker="^GSPC"
    start_date="2022-01-01"
    end_date="2022-12-31"
    period="1mo"
    risk_free_rate="^IRX"
    risk_aversion = compute_risk_aversion(market_ticker, start_date, end_date, period, risk_free_rate)
    assert isinstance(risk_aversion, float), "Risk aversion calculation failed: risk aversion is not a float."
    print(f"Risk aversion computed successfully: {risk_aversion}")

    print("Testing get_market_weights...")
    tickers = ["AAPL", "MSFT", "GOOG"]
    weights = get_market_weights(tickers)
    assert isinstance(weights, np.ndarray), "Market weights should be a numpy array."
    assert np.isclose(weights.sum(), 1.0), "Market weights should sum to 1."
    print(f"Market weights computed successfully: {weights}")

    print("Testing compute_market_implied_return...")
    cov_matrix = np.array([[0.04, 0.02], [0.02, 0.05]])
    weights = np.array([0.6, 0.4])
    risk_aversion = 3.0
    implied_returns = compute_market_implied_return(weights, cov_matrix, risk_aversion)
    assert isinstance(implied_returns, np.ndarray), "Implied returns should be a numpy array."
    print(f"Market implied returns computed successfully: {implied_returns}")

    print("Testing apply_investor_view...")
    market_returns = np.array([0.08, 0.12, 0.10])
    cov_matrix = np.array([[0.04, 0.02, 0.01], [0.02, 0.05, 0.015], [0.01, 0.015, 0.06]])
    p = np.array([[1, -1, 0], [0, 1, -1]]) 
    q = np.array([0.02, 0.01]) 
    omega = np.diag([0.0001, 0.0005])  
    adjusted_returns = apply_investor_view(market_returns, cov_matrix, p, q, omega)
    assert isinstance(adjusted_returns, np.ndarray), "Adjusted returns should be a numpy array."
    print(f"Adjusted returns computed successfully: {adjusted_returns}")

    print("Testing black_litterman...")
    adjusted_returns = black_litterman(tickers=tickers, risk_free_rate="^IRX", cov_matrix=cov_matrix, p=p, q=q, omega=omega, market_ticker="^GSPC", start_date="2022-01-01", end_date="2023-12-01", period="1d")
    assert isinstance(adjusted_returns, np.ndarray), "Adjusted returns should be a numpy array."
    print(f"Adjusted returns (Black-Litterman) computed successfully: {adjusted_returns}")

def run_test():
    cov_matrix, risk_free_rate = test_data()
    test_optimization(cov_matrix, risk_free_rate)
    test_blacklitterman()

run_test()