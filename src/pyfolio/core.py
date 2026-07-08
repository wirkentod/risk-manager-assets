"""Core functions: load data and compute correlations."""
from pathlib import Path
import pandas as pd
import numpy as np


def load_data(path):
    """Load CSV time series into a DataFrame.

    Expects a CSV with one column per asset. If a `Date` or similar index column
    is present it will be preserved but only numeric columns are used for
    correlation calculation.
    """
    p = Path(path)
    df = pd.read_csv(p)
    return df

def compute_risk(df, term, dailyreturn,periods):
    """Compute risk metrics for a portfolio."""
    dr = compute_daily_return(df, term, dailyreturn)
    #calculate return and risk for each asset
    returns = dr.mean() * periods # Annualized return
    risks = dr.std() * np.sqrt(periods) # Annualized volatility
    sharpe_ratios = returns / risks # Sharpe ratio
    print(f"Return Ordered:\n {returns.sort_values(ascending=False)}")
    print(f"Risks Ordered:\n {risks.sort_values(ascending=True)}")
    print(f"Sharpe Ratios Ordered:\n {sharpe_ratios.sort_values(ascending=False)}")
    #read weights from a file or define them here
    #weight = pd.Series([0.60, 0.40], index=["A", "B"]) 
    weight = pd.Series(
        [0.2597, 0.1782, 0.1429, 0.1074, 0.0943, 0.0725, 0.0368, 0.0364, 0.036, 0.0357],
        index = ["ORYGENC1","FERREYC1","MINSURI1","IPCHBC1","BBVAC1","ALICORC1","BACKUSI1","CARTAVC1","SNJUANI1","SIDERC1"]
    )
    #returnP = weight.dot(dr.mean()) * periods
    returnP = np.sum(weight * dr.mean()) * periods # Annualized portfolio return
    riskP = np.sqrt(weight.dot(dr.cov()).dot(weight)) * np.sqrt(periods) # Annualized portfolio volatility
    sharpeP = returnP / riskP # Portfolio Sharpe ratio
    print(f"Return portfolio: {returnP}")   
    print(f"Risk portfolio: {riskP}")
    print(f"Sharpe ratio portfolio: {sharpeP}")

    #corr_matrix = dr.corr()
    #risk_portfolio = np.sqrt(weight.dot(cov_matrix).dot(weight))
    return riskP

def compute_daily_return(df, term, dailyreturn):
    """Compute a term daily return for a portfolio."""
    term_choice = {"1W": 3, "1M": 21, "2M": 42, "3M": 63, "1A": 252}
    numeric = df.select_dtypes(include="number")[:term_choice[term]+1]
    if dailyreturn == "log":
        return np.log(numeric / numeric.shift(-1))
    elif dailyreturn == "simple":
        return (numeric/numeric.shift(-1)-1)
    else:
        raise ValueError("Invalid daily return method. Choose 'log' or 'simple'.")
 
def compute_correlation(df, term, dailyreturn, method="pearson"):
    """Compute correlation matrix for numeric columns.

    method: 'pearson', 'spearman', 'kendall'
    """
    dr = compute_daily_return(df, term, dailyreturn)
    return dr.corr(method=method)


def save_corr(df_corr, out_path):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df_corr.to_csv(p)
