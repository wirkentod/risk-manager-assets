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
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    pfolio_assets: list, 
    pfolio_weights: list,
    anualperiod: int,
    riskfreerate: float
) -> tuple:
    """Compute risk metrics for a portfolio.
    """
    # Read weights from a file or define them here
    weight = pd.Series(pfolio_weights, index = pfolio_assets).values
    return portfolio_stats(weight, expected_returns, cov_matrix, anualperiod, riskfreerate) #returnP, riskP, sharpeP

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

def compute_mean(
    dailyreturn: pd.DataFrame
) -> pd.DataFrame:
    """Compute mean for numeric columns.
    """
    return dailyreturn.mean()

def portfolio_stats(
    weights: np.ndarray, 
    expected_returns: np.ndarray, 
    cov_matrix: np.ndarray,
    anualperiod: int, 
    riskfreerate: float
) -> tuple:
    """Computes exact portfolio statistics using optimized linear algebra."""
    p_return = compute_return(expected_returns, weights, anualperiod)
    p_risk = compute_risk(cov_matrix, weights, anualperiod)
    p_sharpe = compute_sharpe_ratio(p_return, p_risk, riskfreerate)
    return p_return, p_risk, p_sharpe

def compute_return(expected_return: np.ndarray, weights: np.ndarray, anualperiod: int) -> float:
    # Point product between weigths and expected returns
    return np.dot(weights, expected_return) * anualperiod # Annualized return ratio

def compute_risk(cov_matrix: np.ndarray, weights: np.ndarray, anualperiod: int) -> float:
    # Fast matricial multiplication (w^T * Cov * w)
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(anualperiod) # Annualized risk ratio

def compute_sharpe_ratio(returnP: float, riskP: float, riskfreerate: float) -> float:
    return (returnP - riskfreerate) / riskP

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
    
    # --- 1. PRECOMPUTAR (SOLUCIÓN AL CUELLO DE BOTELLA) ---
    # Se calcula una sola vez fuera de los optimizadores
    asset_returns = dailyreturn[pfolio_assets].mean() * anualperiod
    cov_matrix = dailyreturn[pfolio_assets].cov() * anualperiod
    
    num_assets = len(pfolio_assets)
    # Pasamos datos estáticos (retornos y covarianza) en lugar del DataFrame completo
    optimization_args = (asset_returns.values, cov_matrix.values, anualperiod, riskfreerate)
    
    # --- CONSTRAINTS AND BOUNDS ---
    sum_weights_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_weights = np.array(num_assets * [1.0 / num_assets])

    # =========================================================================
    # OPTIMIZATION 1: MAXIMIZE SHARPE RATIO
    # =========================================================================
    opt_sharpe = sco.minimize(
        fun=negate_sharpe, # Debe aceptar (w, asset_returns, cov_matrix, rf)
        x0=init_weights,
        args=optimization_args,
        method='SLSQP',
        bounds=bounds,
        constraints=[sum_weights_constraint]
    )
    optimal_weights = opt_sharpe['x']
    opt_ret, opt_vol, opt_sh = portfolio_stats(optimal_weights, *optimization_args)
    optimal_portfolio = pd.Series({'Return': opt_ret, 'Risk': opt_vol, 'SharpeRatio': opt_sh})

    # =========================================================================
    # OPTIMIZATION 2: EFFICIENT FRONTIER CURVE MAPPING
    # =========================================================================
    opt_min_vol = sco.minimize(
        fun=minimize_volatility, # Debe aceptar (w, asset_returns, cov_matrix, rf)
        x0=init_weights,
        args=optimization_args,
        method='SLSQP',
        bounds=bounds,
        constraints=[sum_weights_constraint]
    )
    
    min_return_boundary = portfolio_stats(opt_min_vol['x'], *optimization_args)[0]
    max_return_boundary = asset_returns.max()
    
    target_returns = np.linspace(min_return_boundary, max_return_boundary, 20)
    frontier_vols = []
    transition_weights_list = []

    # El bucle ahora es drásticamente más rápido porque las funciones internas solo hacen álgebra lineal básica
    for target in target_returns:
        frontier_constraints = [
            sum_weights_constraint,
            # Producto punto rápido en lugar de llamar a una función pesada
            {'type': 'eq', 'fun': lambda w, r=asset_returns.values: np.dot(w, r) - target}
        ]
        
        res = sco.minimize(
            fun=minimize_volatility,
            x0=init_weights,
            args=optimization_args,
            method='SLSQP',
            bounds=bounds,
            constraints=frontier_constraints
        )
        
        frontier_vols.append(res['fun'])
        transition_weights_list.append(res['x'])

    # --- CONSOLIDATE DATA ---
    efficient_frontier_points = pd.DataFrame({
        'Return': target_returns,
        'Risk': frontier_vols,
        'SharpeRatio': (target_returns - riskfreerate) / np.array(frontier_vols)
    })

    transition_map_points = pd.DataFrame(
        transition_weights_list,
        columns=pfolio_assets,
        index=frontier_vols
    )

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

def minimize_volatility(
    weights: np.ndarray, 
    asset_returns: pd.DataFrame, 
    cov_matrix: np.ndarray,
    anualperiod: int,
    riskfreerate: float
) -> float:
    """Objective function to minimize Volatility."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def negate_sharpe(
    weights: np.ndarray, 
    asset_returns: pd.DataFrame, 
    cov_matrix: np.ndarray, 
    anualperiod: int,
    riskfreerate: float
) -> float:
    """Objective function to maximize Sharpe Ratio."""
    return -portfolio_stats(weights, asset_returns, cov_matrix, anualperiod, riskfreerate)[2]

def compute_pca(
    dailyreturn: pd.DataFrame, 
    anualperiod: int, 
    pfolio_assets: list
) -> tuple:
    """Compute PCA for dimensionality reduction."""
    #clean the data by dropping rows with NaN values for the selected assets
    dr_filtered = dailyreturn[pfolio_assets].dropna()

    pass  # Placeholder for PCA implementation

