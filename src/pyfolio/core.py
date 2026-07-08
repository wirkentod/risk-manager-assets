"""Core functions: load data and compute correlations."""
from pathlib import Path
import pandas as pd
import numpy as np


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
    #print(f"Return portfolio: {returnP}")   
    #print(f"Risk portfolio: {riskP}")
    #print(f"Sharpe ratio portfolio: {sharpeP}")
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
