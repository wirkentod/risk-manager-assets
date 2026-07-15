"""Core functions: load data and compute correlations."""
from pathlib import Path
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import scipy.optimize as sco

# -----------------------------------------------------------------------------------
# Author: Christian Quispe
# Date: 08/07/26
# Description: This project involves emerging markets portfolio optimization using 
#              Monte Carlo simulation that generates random portfolios and evaluates 
#              their performance based on risk and return metrics. The goal is to identify
#              the optimal portfolio allocation that maximizes returns while minimizing risk, 
#              in line with Modern Portfolio Theory and the Efficient Frontier concept.
# -----------------------------------------------------------------------------------

def load_data(path:str, assets: list) -> pd.DataFrame:
    """Load CSV time series into a DataFrame.

    Expects a CSV with one column per asset. If a `Date` or similar index column
    is present it will be preserved but only numeric columns are used for
    correlation calculation.
    """
    p = Path(path)
    df = pd.read_csv(p)
    # Logic to filter only the columns that are in the assets list including
    return df[assets]

def compute_portfolio_metrics(
    dailyreturn: pd.DataFrame, 
    anualperiod: int, 
    riskfreerate: float, 
    pfolio_assets: list, 
    pfolio_weights: list
) -> tuple:
    """Compute risk metrics for a portfolio.
    """
    # Read weights from a file or define them here
    weight = pd.Series(pfolio_weights, index = pfolio_assets)
    return portfolio_stats(weight, dailyreturn, anualperiod, riskfreerate) #returnP, riskP, sharpeP

def compute_assets_metrics(
    dailyreturn: pd.DataFrame, 
    anualperiod: int, 
    riskfreerate: float
) -> pd.DataFrame:
    """Compute risk metrics for each asset.
    """
    # Compute return, risk and sharpe ratio for each asset
    returns = dailyreturn.mean() * anualperiod # Annualized return
    risks = dailyreturn.std() * np.sqrt(anualperiod) # Annualized volatility
    sharpe_ratios = (returns - riskfreerate) / risks # Annualized Sharpe ratio
    print(f"Return Ordered:\n {returns.sort_values(ascending=False)}")
    print(f"Risks Ordered:\n {risks.sort_values(ascending=True)}")
    print(f"Sharpe Ratios Ordered:\n {sharpe_ratios.sort_values(ascending=False)}")
    # DataFrame with assets metrics
    return pd.DataFrame({
        'Return': returns,
        'Risk': risks,
        'SharpeRatio': sharpe_ratios
    })

def compute_daily_return(
    datafolio: pd.DataFrame, 
    term: str, 
    dailychange: str
) -> pd.DataFrame:
    """Compute a term daily return for a portfolio.
    term: '1W', '1M', '2M', '3M', '6M', '1A', '2A'.
    dailychange: 'log', 'simple'
    """
    term_choice = {"1W": 5, "1M": 21, "2M": 42, "3M": 63, "6M": 126, "1A": 252, "2A": 504}
    numeric = datafolio.select_dtypes(include="number")[:term_choice[term]+1]
    if dailychange == "log":
        return np.log(numeric / numeric.shift(-1))
    elif dailychange == "simple":
        return (numeric/numeric.shift(-1)-1)
    else:
        raise ValueError("Invalid daily return method. Choose 'log' or 'simple'.")

def standarized_daily_return(dailyreturn: pd.DataFrame) -> pd.DataFrame:
    return (dailyreturn - dailyreturn.mean()) / dailyreturn.std()

def compute_correlation(
    dailyreturn: pd.DataFrame, 
    method="pearson"
) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns.
    """
    return dailyreturn.corr(method=method)

def compute_covariance(
    dailyreturn: pd.DataFrame
) -> pd.DataFrame:
    """Compute covariance matrix for numeric columns.
    """
    return dailyreturn.cov()

def portfolio_stats(
    weights: np.ndarray, 
    dailyreturn: pd.DataFrame, 
    anualperiod: int, 
    riskfreerate: float
) -> tuple:
    """Computes exact portfolio statistics using external helpers."""
    p_return = compute_return(dailyreturn, weights, anualperiod)
    p_risk = compute_risk(dailyreturn, weights, anualperiod)
    p_sharpe = compute_sharpe_ratio(p_return, p_risk, riskfreerate)
    return p_return, p_risk, p_sharpe

def compute_return(dailyreturn: pd.DataFrame, weights: np.ndarray, anualperiod: int) -> float:
    return np.sum(weights * dailyreturn.mean()) * anualperiod # Annualized return ratio

def compute_risk(dailyreturn: pd.DataFrame, weights: np.ndarray, anualperiod: int) -> float:
    return np.sqrt(weights.dot(dailyreturn.cov()).dot(weights)) * np.sqrt(anualperiod) # Annualized risk ratio

def compute_sharpe_ratio(returnP: float, riskP: float, riskfreerate: float) -> float:
    return (returnP - riskfreerate) / riskP # Annualized sharpe ratio

def save_corr(df_corr: pd.DataFrame, out_path: str):
    """Save correlation matrix to a CSV file."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df_corr.to_csv(p)

def compute_efficient_frontier(
    dailyreturn: pd.DataFrame, 
    anualperiod: int, 
    riskfreerate: float, 
    pfolio_assets: list
) -> tuple:
    """
    Mathematically computes optimal portfolio weights and the efficient frontier curve
    using externalized objective functions.
    """
    num_assets = len(pfolio_assets)
    
    # Tuples containing external variables needed by the functions
    optimization_args = (dailyreturn, anualperiod, riskfreerate)

    # --- OPTIMIZER CONSTRAINTS AND BOUNDS ---
    sum_weights_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_weights = np.array(num_assets * [1.0 / num_assets])

    # =========================================================================
    # OPTIMIZATION 1: MAXIMIZE SHARPE RATIO
    # =========================================================================
    opt_sharpe = sco.minimize(
        fun=negate_sharpe, 
        x0=init_weights, 
        args=optimization_args,  # <-- ARGS PASSED HERE
        method='SLSQP', 
        bounds=bounds, 
        constraints=[sum_weights_constraint]
    )
    
    optimal_weights = opt_sharpe['x']
    opt_ret, opt_vol, opt_sh = portfolio_stats(optimal_weights, *optimization_args)
    
    optimal_portfolio = pd.Series({
        'Return': opt_ret, 'Risk': opt_vol, 'SharpeRatio': opt_sh
    })

    # =========================================================================
    # OPTIMIZATION 2: EFFICIENT FRONTIER CURVE MAPPING
    # =========================================================================
    opt_min_vol = sco.minimize(
        fun=minimize_volatility, 
        x0=init_weights, 
        args=optimization_args,  # <-- ARGS PASSED HERE
        method='SLSQP', 
        bounds=bounds, 
        constraints=[sum_weights_constraint]
    )
    
    # Extract the numeric values for the boundaries
    min_return_boundary = portfolio_stats(opt_min_vol['x'], *optimization_args)[0]
    max_return_boundary = (dailyreturn.mean() * anualperiod).max()

    target_returns = np.linspace(min_return_boundary, max_return_boundary, 20)
    frontier_vols = []
    transition_weights_list = []

    for target in target_returns:
        # Extra constraint: portfolio return must match target
        frontier_constraints = [
            sum_weights_constraint,
            {'type': 'eq', 'fun': lambda w: portfolio_stats(w, *optimization_args)[0] - target}
        ]
        
        res = sco.minimize(
            fun=minimize_volatility, 
            x0=init_weights, 
            args=optimization_args,  # <-- ARGS PASSED HERE
            method='SLSQP', 
            bounds=bounds, 
            constraints=frontier_constraints
        )
        # Save volatility metric
        frontier_vols.append(res['fun'])
        # Save exact optimized array of weights for this specific risk level
        transition_weights_list.append(res['x'])

    efficient_frontier_points = pd.DataFrame({
        'Return': target_returns,
        'Risk': frontier_vols,
        'SharpeRatio': compute_sharpe_ratio(target_returns, np.array(frontier_vols), riskfreerate)
    })

    # --- CONSOLIDATE TRANSITION MAP DATA ---
    # DataFrame rows = Risk levels (X), columns = Assets, values = allocation weights (Y)
    transition_map_points = pd.DataFrame(
        transition_weights_list, 
        columns=pfolio_assets, 
        index=frontier_vols  # Set the risk/volatility as the index for easier plotting
    )

    # --- CONSOLE REPORTING ---
    print(f"Optimal Portfolio (Exact):\n{optimal_portfolio}\n")
    print(f"Optimal Weights (Exact):\n{pd.Series(optimal_weights, index=pfolio_assets).sort_values(ascending=False)}")
    return optimal_weights, optimal_portfolio, efficient_frontier_points, transition_map_points 

def compute_montecarlo_simulation(
    dailyreturn: pd.DataFrame, 
    anualperiod: int, 
    riskfreerate: float, 
    pfolio_assets: list, 
    num_simulations: int
) -> tuple:
    """Compute Monte Carlo simulation for portfolio optimization.
    anualperiod: number of trading days in a year (e.g., 252).
    pfolio_assets: list of asset names.
    num_simulations: number of random portfolios to simulate.
    """
    # MonteCarlo Simulation
    results = np.zeros((4, num_simulations))
    weights_record = np.zeros((len(pfolio_assets), num_simulations))
    alpha_param = 0.1 # Coefficient for the Dirichlet distribution to ensure weights sum to 1 and are non-negative
    for i in range(num_simulations):
        # Random weights using Dirichlet distribution to ensure they sum to 1 and are non-negative
        weights = np.random.dirichlet([alpha_param] * len(pfolio_assets))
        weights_record[:, i] = weights
        # Annualized portfolio return
        portfolio_return = compute_return(dailyreturn, weights, anualperiod) 
        # Annualized portfolio volatility
        portfolio_stddev = compute_risk(dailyreturn, weights, anualperiod)
        # Annualized Sharpe ratio
        portfolio_sharperatio = compute_sharpe_ratio(portfolio_return, portfolio_stddev, riskfreerate)

        results[0, i] = portfolio_return
        results[1, i] = portfolio_stddev
        results[2, i] = portfolio_sharperatio
        results[3, i] = i  # Index of the simulation
        
    # Convert results to a DataFrame
    simulated_portfolios = pd.DataFrame(results.T, columns=['Return', 'Risk', 'SharpeRatio', 'Simulation'])

    # Find the portfolio with the highest Sharpe ratio
    optimal_idx = simulated_portfolios['SharpeRatio'].idxmax()
    optimal_portfolio = simulated_portfolios.loc[optimal_idx]
    optimal_weights = weights_record[:, optimal_idx]

    print(f"Optimal Portfolio:\n{optimal_portfolio}")
    print(f"Optimal Weights:\n{pd.Series(optimal_weights, index=pfolio_assets).sort_values(ascending=False)}")
    return optimal_weights, optimal_portfolio, simulated_portfolios

def negate_sharpe(
    weights: np.ndarray, 
    dailyreturn: pd.DataFrame, 
    anualperiod: int, 
    riskfreerate: float
) -> float:
    """Objective function to maximize Sharpe Ratio."""
    return -portfolio_stats(weights, dailyreturn, anualperiod, riskfreerate)[2]

def minimize_volatility(
    weights: np.ndarray, 
    dailyreturn: pd.DataFrame, 
    anualperiod: int, 
    riskfreerate: float
) -> float:
    """Objective function to minimize Volatility."""
    return portfolio_stats(weights, dailyreturn, anualperiod, riskfreerate)[1]

def compute_pca(
    dailyreturn: pd.DataFrame, 
    anualperiod: int, 
    pfolio_assets: list
) -> tuple:
    """Compute PCA for dimensionality reduction."""
    #compute covariance matrix and annualize it
    #cov_matrix_annualized = compute_covariance(dailyreturn) * anualperiod
    #corr_matrix = compute_correlation(dailyreturn)
    dailyreturnstandarized = standarized_daily_return(dailyreturn)
    #only if we consider 3 vector risk: (a) market/beta, (b) sectorial y (c) money rotation
    #pca = PCA(n_components=min(3, len(pfolio_assets)))
    pca = PCA(n_components=len(pfolio_assets))
    #compute eigenvalues and eigenvectors
    pca.fit(dailyreturnstandarized)
    #eigenvalues (type: np.ndarray), variance explained by each component:
    eigenvalues = pca.explained_variance_ratio_
    #eigenvector (type: pd.DataFrame), load factors, relation between each asset and component
    eigenvectors = pd.DataFrame(
        pca.components_.T, 
        columns=[f'PC{i+1}' for i in range(pca.n_components_)], 
        index=pfolio_assets
    )
    return eigenvalues, eigenvectors

