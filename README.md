# Portfolio Optimization – Markowitz Model

## Introduction

This project implements the **Markowitz Mean-Variance Optimization Model**, allowing users to build optimal portfolios based on their risk-return preferences. It supports key features such as:

- Efficient frontier computation
- Maximum Sharpe Ratio portfolio
- Optional **short selling**
- Optional **Black-Litterman model** extension
- Visual representation of results

It is suitable for financial analysis, backtesting, or learning portfolio theory through hands-on Python usage.

---

## Simplified Mathematical Overview

The Markowitz model aims to find the best portfolio allocation for a given level of risk. Key components include:

- **Expected return** of the portfolio:
  \[
  \mu_p = \sum_{i=1}^{n} w_i \mu_i
  \]
  where \( w_i \) is the weight of asset \( i \), and \( \mu_i \) is its expected return.

- **Portfolio variance (risk)**:
  \[
  \sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \sigma_{ij}
  \]
  where \( \sigma_{ij} \) is the covariance between assets \( i \) and \( j \).

- **Sharpe Ratio**:
  \[
  S = \frac{\mu_p - r_f}{\sigma_p}
  \]
  where \( r_f \) is the risk-free rate. The portfolio that maximizes this ratio is the **tangency portfolio**.

The set of all optimal portfolios forms the **efficient frontier**, which defines the best trade-offs between return and risk.

---

## Usage

### Requirements
- Python 3.8+
- Common packages: `pandas`, `numpy`, `matplotlib`, `yfinance`, etc.

Install dependencies (if not already):
```bash
pip install -r requirements.txt
```

### Project Structure
- `__main__.py` – main script (contains sample usage)
- `data.py` – handles data loading, cleaning, and validation
- `optimization.py` – portfolio optimization logic
- `visualization.py` – efficient frontier and graph plotting
- `blacklitterman.py` – Black-Litterman model support

### ▶Running the Example

```bash
python __main__.py
```

This will:
- Load asset price data from `test_dataset.csv`
- Compute the efficient frontier
- Print the optimized weights

---

## Key Functions

### `efficient_frontier(data, precision=50, ...)`
- Computes the efficient frontier
- Can display a graph (with Capital Market Line)
- Supports short selling and Sharpe optimization

### `min_variance_portfolio(data, ...)`
- Returns the **minimum variance** portfolio

### `max_sharpe_portfolio(data, risk_free_rate, ...)`
- Returns the portfolio with the **maximum Sharpe Ratio**

### `load_data(...)` and `get_risk_free_rate(...)`
- Fetch data via Yahoo Finance for selected tickers and risk-free assets

---

## Example

```python
from __main__ import load_data, efficient_frontier

data = load_data(tickers=["AAPL", "MSFT", "GOOGL"])
results = efficient_frontier(data, show_graph=True)
print(results["optimized weights"])
```

---

## Notes

- The model supports the **Black-Litterman framework** to incorporate subjective market views.
- The code is modular and extensible for research or production applications.
- Ideal for quantitative finance learners and practitioners.

---

## License

This project is open-source and free to use for educational and research purposes.