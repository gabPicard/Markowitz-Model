import matplotlib.pyplot as plt
import numpy as np

def plot_efficient_frontier(frontier, max_sharpe=None, min_variance=None, risk_free_rate = None, title="Efficient frontier"):
    """
    Plot the Efficient Frontier and optional Capital Market Line (CML).

    This function visualizes the Efficient Frontier of a portfolio, plots specific 
    points such as the minimum variance portfolio and the maximum Sharpe ratio portfolio, 
    and optionally overlays the Capital Market Line (CML) if a risk-free rate is provided.

    Parameters:
    - frontier: list of tuples
        A list of (risk, return) tuples representing the Efficient Frontier. 
        Each tuple contains the portfolio's standard deviation (risk) and expected return.
    
    - max_sharpe: tuple of floats, optional
        A tuple (risk, return) representing the portfolio with the maximum Sharpe ratio.
        Default is None. If provided, it will be used to calculate the CML slope.

    - min_variance: tuple of floats, optional
        A tuple (risk, return) representing the minimum variance portfolio. 
        Default is None.

    - risk_free_rate: float, optional
        The risk-free rate used to calculate and plot the Capital Market Line (CML).
        Default is None. If not provided, the CML will not be plotted.

    - title: str, optional
        The title of the plot. Default is "Efficient frontier".

    Returns:
    - None
        Displays a Matplotlib plot showing the Efficient Frontier, CML (if applicable), 
        and the key points like min-variance and max-Sharpe portfolios.

    """
    risks, returns = zip(*frontier)

    plt.figure(figsize=(10,6))
    plt.plot(risks, returns, label="Efficient frontier", color="blue", marker="o", linestyle="-")
    
    if min_variance:
        plt.scatter(min_variance[0], min_variance[1], color="green", label="Min Variance", zorder=5)
    
    if max_sharpe:
        max_sharpe_risk, max_sharpe_return = max_sharpe
    sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_risk
    
    cml_risks = np.linspace(0, max(risks) * 1.2, 100)
    cml_returns = risk_free_rate + sharpe_ratio * cml_risks
    
    plt.plot(cml_risks, cml_returns, label="Capital Market Line (CML)", color="red", linestyle="--")

    plt.title(title)
    plt.xlabel("Risk (Standard Deviation)")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.show()

