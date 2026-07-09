"""Core functions: load data and compute correlations."""
from pathlib import Path
import pandas as pd
import numpy as np
# -----------------------------------------------------------------------------------
# Author: Christian Quispe
# Date: 08/07/26
# Description: This project involves emerging markets portfolio optimization using 
#              Monte Carlo simulation that generates random portfolios and evaluates 
#              their performance based on risk and return metrics. The goal is to identify
#              the optimal portfolio allocation that maximizes returns while minimizing risk, 
#              in line with Modern Portfolio Theory and the Efficient Frontier concept.
# -----------------------------------------------------------------------------------

def load_data(path:str) -> pd.DataFrame:
    """Load CSV time series into a DataFrame.

    Expects a CSV with one column per asset. If a `Date` or similar index column
    is present it will be preserved but only numeric columns are used for
    correlation calculation.
    """
    p = Path(path)
    df = pd.read_csv(p)
    return df

def compute_portfolio_metrics(df: pd.DataFrame, term: str, dailyreturn: int, anualperiod: int, pfolio_assets: list, pfolio_weights: list) -> tuple:
    """Compute risk metrics for a portfolio.
    term: '1W', '1M', '2M', '3M', '1A'.
    dailyreturn: 'log', 'simple'
    """
    dr = compute_daily_return(df, term, dailyreturn)
    #calculate return, risk and sharpe ratio for each asset
    returns = (1 + dr.mean()) ** anualperiod - 1 # Annualized return
    risks = dr.std() * np.sqrt(anualperiod) # Annualized volatility
    sharpe_ratios = returns / risks # Sharpe ratio
    print(f"Return Ordered:\n {returns.sort_values(ascending=False)}")
    print(f"Risks Ordered:\n {risks.sort_values(ascending=True)}")
    print(f"Sharpe Ratios Ordered:\n {sharpe_ratios.sort_values(ascending=False)}")
    #read weights from a file or define them here
    weight = pd.Series(pfolio_weights, index = pfolio_assets)
    #compute portfolio return, risk and sharpe ratio
    returnP = (1 + np.sum(weight * dr.mean()) ) ** anualperiod - 1 # Annualized portfolio return ratio
    riskP = np.sqrt(weight.dot(dr.cov()).dot(weight)) * np.sqrt(anualperiod) # Annualized portfolio risk ratio
    sharpeP = returnP / riskP # Annualized portfolio sharpe ratio
    return returnP, riskP, sharpeP

def compute_daily_return(df: pd.DataFrame, term: str, dailyreturn: str) -> pd.DataFrame:
    """Compute a term daily return for a portfolio.
    term: '1W', '1M', '2M', '3M', '1A'.
    dailyreturn: 'log', 'simple'
    """
    term_choice = {"1W": 3, "1M": 21, "2M": 42, "3M": 63, "1A": 252}
    numeric = df.select_dtypes(include="number")[:term_choice[term]+1]
    if dailyreturn == "log":
        return np.log(numeric / numeric.shift(-1))
    elif dailyreturn == "simple":
        return (numeric/numeric.shift(-1)-1)
    else:
        raise ValueError("Invalid daily return method. Choose 'log' or 'simple'.")
 
def compute_correlation(df: pd.DataFrame, term: str, dailyreturn: str, method="pearson") -> pd.DataFrame:
    """Compute correlation matrix for numeric columns.
    term: '1W', '1M', '2M', '3M', '1A'.
    dailyreturn: 'log', 'simple'.
    method: 'pearson', 'spearman', 'kendall'
    """
    dr = compute_daily_return(df, term, dailyreturn)
    return dr.corr(method=method)


def save_corr(df_corr: pd.DataFrame, out_path: str):
    """Save correlation matrix to a CSV file."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df_corr.to_csv(p)

def compute_montecarlo_simulation(df: pd.DataFrame, term: str, dailyreturn: str, anualperiod: int, riskfreerate: float, pfolio_assets: list, num_simulations: int) -> tuple:
    """Compute Monte Carlo simulation for portfolio optimization.
    term: '1W', '1M', '2M', '3M', '1A'.
    dailyreturn: 'log', 'simple'.
    anualperiod: number of trading days in a year (e.g., 252).
    pfolio_assets: list of asset names.
    num_simulations: number of random portfolios to simulate.
    """
    #Daily return Portfolio
    dr = compute_daily_return(df, term, dailyreturn)
    #MonteCarlo Simulation
    results = np.zeros((4, num_simulations))
    weights_record = np.zeros((len(pfolio_assets), num_simulations))
    for i in range(num_simulations):
        #Random weights
        weights = np.random.random(len(pfolio_assets))
        weights /= np.sum(weights)
        weights_record[:, i] = weights

        #Annualized portfolio return
        portfolio_return = (1 + np.sum(weights * dr.mean()) ) ** anualperiod - 1
        #Annualized portfolio volatility
        portfolio_stddev = np.sqrt(weights.dot(dr.cov()).dot(weights)) * np.sqrt(anualperiod)
        #Annualized Sharpe ratio
        portfolio_sharperatio = (portfolio_return - riskfreerate) / portfolio_stddev

        results[0, i] = portfolio_return
        results[1, i] = portfolio_stddev
        results[2, i] = portfolio_sharperatio
        results[3, i] = i  # Index of the simulation
        
    #Convert results to a DataFrame
    simulated_portfolios = pd.DataFrame(results.T, columns=['Return', 'Risk', 'SharpeRatio', 'Simulation'])

    #Find the portfolio with the highest Sharpe ratio
    optimal_idx = simulated_portfolios['SharpeRatio'].idxmax()
    optimal_portfolio = simulated_portfolios.loc[optimal_idx]
    optimal_weights = weights_record[:, optimal_idx]

    print(f"Optimal Portfolio:\n{optimal_portfolio}")
    print(f"Optimal Weights:\n{pd.Series(optimal_weights, index=pfolio_assets).sort_values(ascending=False)}")
    return optimal_weights, optimal_portfolio, simulated_portfolios
