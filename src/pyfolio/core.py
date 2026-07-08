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

def compute_risk(df, dailyreturn, term="3M"):
    """Compute risk metrics for a portfolio."""
    dr = compute_daily_return(df, dailyreturn)
    #read weights from a file or define them here
    weight = pd.Series([0.60, 0.40], index=["A", "B"]) 
    riskP = np.sqrt(weight.dot(dr.cov()).dot(weight))
    print(f"Risk portfolio: {riskP}")
    #corr_matrix = dr.corr()
    #risk_portfolio = np.sqrt(weight.dot(cov_matrix).dot(weight))
    return riskP

def compute_daily_return(df, dailyreturn="log"):
    """Compute daily return for a portfolio."""
    numeric = df.select_dtypes(include="number")
    if dailyreturn == "log":
        return np.log(numeric / numeric.shift(1))
    elif dailyreturn == "simple":
        return (numeric/numeric.shift(-1)-1)
    else:
        raise ValueError("Invalid daily return method. Choose 'log' or 'simple'.")
 
def compute_correlation(df, method="pearson"):
    """Compute correlation matrix for numeric columns.

    method: 'pearson', 'spearman', 'kendall'
    """
    numeric = df.select_dtypes(include="number")
    return numeric.corr(method=method)


def save_corr(df_corr, out_path):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df_corr.to_csv(p)
