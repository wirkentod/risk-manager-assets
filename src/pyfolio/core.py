"""Core functions: load data and compute correlations."""
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import scipy.stats as stats

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
    pfolio_weights: list,
    confidencelevel: float,
    numsimulations: int
) -> dict:
    """Compute risk metrics for a portfolio.
    """
    # Read weights from a file or define them here
    weight = pd.Series(pfolio_weights, index = pfolio_assets)
    #must return a dict portfolio
    returnP, riskP, sharpeP = portfolio_stats(weight, dailyreturn, anualperiod, riskfreerate)
    tailriskmetrics = portfolio_valueatrisk(weight, dailyreturn,anualperiod,confidencelevel,numsimulations)
    portfolio = {
        'Return': returnP, 
        'Risk': riskP, 
        'SharpeRatio': sharpeP, 
        'VaR_Hist': tailriskmetrics['VaR_Hist'], 
        'CVaR_Hist': tailriskmetrics['CVaR_Hist'],
        'VaR_Param': tailriskmetrics['VaR_Param'], 
        'CVaR_Param': tailriskmetrics['CVaR_Param'],
        'VaR_MC': tailriskmetrics['VaR_MC'], 
        'CVaR_MC': tailriskmetrics['CVaR_MC']
    }
    return portfolio

def compute_assets_metrics(
    dailyreturn: pd.DataFrame, 
    anualperiod: int, 
    riskfreerate: float, 
    pfolio_weights: list,
    confidence_level: float,
    numsimulations: int
) -> pd.DataFrame:
    """Compute risk metrics for each asset.
    """
    # Compute return, risk and sharpe ratio for each asset
    returns = dailyreturn.mean() * anualperiod # Annualized return
    risks = dailyreturn.std() * np.sqrt(anualperiod) # Annualized volatility
    sharpe_ratios = (returns - riskfreerate) / risks # Annualized Sharpe ratio
    # Compute VaR and CVaR for each asset
    tailriskmetrics = compute_assets_valueatrisk(
        dailyreturn, anualperiod, confidence_level, numsimulations
    )
    # DataFrame with assets metrics
    return pd.DataFrame({
        'Weight': pfolio_weights,
        'Return': returns,
        'Risk': risks,
        'SharpeRatio': sharpe_ratios,
        'VaRHist': tailriskmetrics['VaR_Hist'],
        'CVaRHist': tailriskmetrics['CVaR_Hist'],
        'VaRParam': tailriskmetrics['VaR_Param'],
        'CVaRParam': tailriskmetrics['CVaR_Param'],
        'VaRMC': tailriskmetrics['VaR_MC'],
        'CVaRMC': tailriskmetrics['CVaR_MC']
    })

def compute_assets_valueatrisk(
    dailyreturn: pd.DataFrame, 
    anualperiod: int, 
    confidencelevel: float,
    numsimulations: int
) -> dict:
    # Compute VaR and CVaR historic for each asset
    alpha = 1 - confidencelevel
    # Compute parameter to VaR and CVaR Parametic method
    z_score = stats.norm.ppf(confidencelevel)
    sqrt_time = np.sqrt(anualperiod)
    tailriskresult = {
        'VaR_Hist': {}, 'CVaR_Hist': {},
        'VaR_Param': {}, 'CVaR_Param': {},
        'VaR_MC': {}, 'CVaR_MC': {}
    }
    #Set seed for reproducibility in Monte Carlo
    np.random.seed(42)
    for asset in dailyreturn.columns:
        # Daily parameters baseline by asset
        daily_mean = dailyreturn[asset].mean()
        daily_std = dailyreturn[asset].std()
        # =========================================================================
        # 1. HISTORIC METHOD
        # =========================================================================
        # Historic VaR: lost percentil
        asset_var_hist = -np.percentile(dailyreturn[asset], alpha * 100)
        # Historic CVaR mean return below VaR
        asset_cvar_hist = -dailyreturn[asset][dailyreturn[asset] <= -asset_var_hist].mean()
        tailriskresult['VaR_Hist'][asset] = asset_var_hist * sqrt_time # Annualized Historic VaR
        tailriskresult['CVaR_Hist'][asset] = asset_cvar_hist * sqrt_time # Annualized Historic CVaR
        # =========================================================================
        # 2. PARAMETRIC METHOD
        # =========================================================================
        asset_var_param = -daily_mean + (z_score * daily_std)
        asset_cvar_param = -daily_mean + (daily_std * stats.norm.pdf(z_score) / alpha)
        tailriskresult['VaR_Param'][asset] = asset_var_param * sqrt_time
        tailriskresult['CVaR_Param'][asset] = asset_cvar_param * sqrt_time
        # =========================================================================
        # 3. MONTE CARLO METHOD
        # =========================================================================
        dailyreturn_assetsimulation = np.random.normal(daily_mean, daily_std, numsimulations)
        asset_var_mc = -np.percentile(dailyreturn_assetsimulation, alpha * 100)
        asset_cvar_mc = -dailyreturn_assetsimulation[dailyreturn_assetsimulation <= -asset_var_mc].mean()
        tailriskresult['VaR_MC'][asset] = asset_var_mc * sqrt_time
        tailriskresult['CVaR_MC'][asset] = asset_cvar_mc * sqrt_time
    return {k: pd.Series(v) for k, v in tailriskresult.items()}

def portfolio_valueatrisk(
    weights: np.ndarray, 
    dailyreturn: pd.DataFrame, 
    anualperiod: int, 
    confidencelevel: float,
    numsimulations: int
) -> dict:
    alpha = 1 - confidencelevel
    z_score = stats.norm.ppf(confidencelevel)
    sqrt_time = np.sqrt(anualperiod)
    tailriskresult = {
        'VaR_Hist': {}, 'CVaR_Hist': {},
        'VaR_Param': {}, 'CVaR_Param': {},
        'VaR_MC': {}, 'CVaR_MC': {}
    }
    # Portfolio Baseline (daily escale)
    dailyreturn_mean = dailyreturn.mean()
    dailyreturn_cov = dailyreturn.cov()
    folio_mean_daily = np.dot(weights, dailyreturn_mean)
    folio_std_daily = np.sqrt(np.dot(weights.T, np.dot(dailyreturn_cov, weights)))
    # =========================================================================
    # 1. HISTORIC METHOD
    # =========================================================================
    dailyreturnfolio = dailyreturn.dot(weights)
    varfolio_hist = -np.percentile(dailyreturnfolio, alpha * 100)
    cvarfolio_hist = -dailyreturnfolio[dailyreturnfolio <= -varfolio_hist].mean()
    tailriskresult['VaR_Hist']  = varfolio_hist *  sqrt_time # Annualized VaR ratio
    tailriskresult['CVaR_Hist']  = cvarfolio_hist * sqrt_time # Annualized CVaR ratio
    # =========================================================================
    # 2. PARAMETRIC METHOD
    # =========================================================================
    varfolio_param = -folio_mean_daily + (z_score * folio_std_daily)
    cvarfolio_param = -folio_mean_daily + (folio_std_daily * stats.norm.pdf(z_score) / alpha)
    tailriskresult['VaR_Param'] = varfolio_param * sqrt_time
    tailriskresult['CVaR_Param'] = cvarfolio_param * sqrt_time
    # =========================================================================
    # 3. MONTE CARLO METHOD
    # =========================================================================
    np.random.seed(42)
    cholesky_matrix = np.linalg.cholesky(dailyreturn_cov)
    random_normal = np.random.normal(0, 1, size=(len(weights), numsimulations))
    dailyreturn_simulation = np.dot(cholesky_matrix, random_normal).T + dailyreturn_mean.values
    dailyreturn_foliosimulation = np.dot(dailyreturn_simulation, weights)
    varfolio_mc = -np.percentile(dailyreturn_foliosimulation, alpha * 100)
    limitreturns_mc = dailyreturn_foliosimulation[dailyreturn_foliosimulation <= -varfolio_mc]
    cvarfolio_mc = -limitreturns_mc.mean()
    tailriskresult['VaR_MC'] = varfolio_mc * sqrt_time
    tailriskresult['CVaR_MC'] = cvarfolio_mc * sqrt_time
    return tailriskresult

def compute_risk_descomposition(covfolioanual, pfolio_assets, pfolio_weights, riskfolio):
    # Compute %risk by asset
    weights = pd.Series(pfolio_weights, index = pfolio_assets)
    # Compute Marginal Contribution to Risk (MCTR)
    mctr = np.dot(covfolioanual, weights) / riskfolio
    # Compute Absolute Contribution to Risk (ACTR)
    actr = weights * mctr
    # Compute percentage contribution of each action
    riskdescomposition = actr / riskfolio
    # Ratio RiskDescomposition/weight
    ratiodescomposition = riskdescomposition / weights
    return pd.DataFrame({
        'Weight': pfolio_weights,
        'RiskDesc' : riskdescomposition,
        'RDW' : ratiodescomposition
    })

def compute_daily_return(
    datafolio: pd.DataFrame, 
    term: str, 
    dailychange: str
) -> pd.DataFrame:
    """Compute a term daily return for a portfolio.
    term: '1W', '1M', '2M', '3M', '6M', '1A', '2A', '3A', '5A', '6A'.
    dailychange: 'log', 'simple'
    """
    term_choice = {"1W": 5, "1M": 21, "2M": 42, "3M": 63, "6M": 126, "1A": 252, "2A": 504, "3A": 756, "5A": 1260, "6A": 1640}
    numeric = datafolio.select_dtypes(include="number")[:term_choice[term]+1]
    if dailychange == "log":
        return np.log(numeric / numeric.shift(-1))
    elif dailychange == "simple":
        return (numeric/numeric.shift(-1)-1)
    else:
        raise ValueError("Invalid daily return method. Choose 'log' or 'simple'.")

def standarized_daily_return(dailyreturn: pd.DataFrame) -> pd.DataFrame:
    return (dailyreturn - dailyreturn.mean()) / dailyreturn.std()

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
    return np.sqrt(weights.dot(compute_covariance(dailyreturn)).dot(weights)) * np.sqrt(anualperiod) # Annualized risk ratio

def compute_sharpe_ratio(returnP: float, riskP: float, riskfreerate: float) -> float:
    return (returnP - riskfreerate) / riskP # Annualized sharpe ratio

def compute_covariance(
    dailyreturn: pd.DataFrame
) -> pd.DataFrame:
    """Compute covariance matrix for numeric columns.
    """
    return dailyreturn.cov()

def compute_correlation(
    dailyreturn: pd.DataFrame, 
    method="pearson"
) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns.
    """
    return dailyreturn.corr(method=method)

def save_corr(df_corr: pd.DataFrame, out_path: str):
    """Save correlation matrix to a CSV file."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df_corr.to_csv(p)

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
    return optimal_weights, optimal_portfolio, simulated_portfolios

def objective_risk_parity(weights, cov_matrix):
    """
    Objective function to minimize the variance between individual asset risk contributions.
    """
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Marginal Contribution to Risk (MCR)
    marginal_risk = np.dot(cov_matrix, weights) / portfolio_volatility
    
    # Absolute Risk Contribution (RC) of each asset
    component_risk = weights * marginal_risk
    
    # Target risk contribution (Equal risk distribution)
    num_assets = len(weights)
    target_risk = portfolio_volatility / num_assets
    
    # Sum of squared errors between actual and target risk contributions
    error = np.sum((component_risk - target_risk) ** 2)
    return error

def compute_riskparity(dailyreturn: pd.DataFrame):
    """
    Computes optimal Risk Parity weights using a Ledoit-Wolf Shrinkage covariance matrix 
    to handle noise and sample error in historical asset returns.
    """
    num_assets = len(dailyreturn.columns)
    
    # --- SHRINKAGE STEP ---
    # Fit the Ledoit-Wolf shrinkage model on the historical return data
    # This shrinks the sample covariance towards a structured target (constant variance)
    lw_estimator = LedoitWolf().fit(dailyreturn.values)
    cov_matrix_shrunk = lw_estimator.covariance_
    
    # Initial guess: Equal weights distribution (1 / N)
    init_weights = np.ones(num_assets) / num_assets
    
    # Portfolio constraint: Weights must sum up to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Boundary conditions: Long-only portfolio (weights between 0 and 1)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Run the optimization using the shrunk covariance matrix as an argument
    result = sco.minimize(
        objective_risk_parity, 
        init_weights, 
        args=(cov_matrix_shrunk,),  # Using the clean shrunk covariance matrix here
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-9}
    )
    
    if not result.success:
        raise RuntimeError("Optimization failed to converge.")     
    return pd.Series(result.x, index=dailyreturn.columns)

def compute_efficient_frontier(
    dailyreturn: pd.DataFrame, 
    anualperiod: int, 
    riskfreerate: float, 
    pfolio_assets: list,
    confidencelevel: float,
    numsimulations: int
) -> tuple:
    """
    Ultra-optimized version of the efficient frontier computation.
    Minimizes overhead by vectorizing inputs and optimizing objective functions.
    """
    num_assets = len(pfolio_assets)
    
    # 1. PASO ÚNICO EN PANDAS: Extracción inmediata a arrays nativos de C (NumPy)
    # Esto reduce la complejidad de la fase previa a O(T * N^2) de forma limpia
    returns_arr = dailyreturn.values
    mean_returns = np.mean(returns_arr, axis=0) * anualperiod
    cov_matrix = np.cov(returns_arr, rowvar=False) * anualperiod

    # 2. VARIABLES DE INICIALIZACIÓN COMPARTIDAS
    init_weights = np.full(num_assets, 1.0 / num_assets)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # Restricción base: La suma de pesos es igual a 1
    sum_weights_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    # =========================================================================
    # OPTIMIZACIÓN 1: MINIMIZAR VOLATILIDAD GLOBAL (Límite Inferior de la Frontera)
    # =========================================================================
    # Empezamos por aquí porque este portafolio nos da automáticamente el límite 
    # inferior de retorno para la frontera eficiente, ahorrándonos una optimización.
    opt_min_vol = sco.minimize(
        fun=fast_volatility, 
        x0=init_weights, 
        args=(cov_matrix,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=[sum_weights_constraint]
    )
    
    if not opt_min_vol['success']:
        raise ValueError("La optimización de mínima volatilidad falló.")
        
    min_vol_weights = opt_min_vol['x']
    min_return_boundary = min_weights_return = min_vol_weights @ mean_returns
    max_return_boundary = np.max(mean_returns)

    # =========================================================================
    # OPTIMIZACIÓN 2: MAXIMIZE SHARPE RATIO
    # =========================================================================
    # Maximizamos Sharpe usando una función simplificada sin condicionales internos
    opt_sharpe = sco.minimize(
        fun=negate_sharpe, 
        x0=init_weights, 
        args=(mean_returns, cov_matrix, riskfreerate), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=[sum_weights_constraint]
    )
    
    if not opt_sharpe['success']:
        raise ValueError("La optimización del Ratio de Sharpe falló.")
    
    optimal_weights = opt_sharpe['x']
    
    # Estadísticas del portafolio óptimo calculadas con álgebra lineal directa (O(N^2))
    opt_ret = optimal_weights @ mean_returns
    opt_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
    opt_sh = (opt_ret - riskfreerate) / opt_vol

    # =========================================================================
    # OPTIMIZACIÓN 3: MAPEO DE LA FRONTERA EFICIENTE
    # =========================================================================
    # Reducimos los puntos a 15 (suficiente para una curva perfecta visualmente)
    target_returns = np.linspace(min_return_boundary, max_return_boundary, 15)
    
    frontier_vols = []
    valid_returns = []
    transition_weights_list = []

    # Reutilizamos las estructuras de restricciones mutando el objetivo en cada paso
    # Esto evita la penalización por recrear diccionarios y lambdas pesadas en memoria
    for target in target_returns:
        frontier_constraints = [
            sum_weights_constraint,
            {'type': 'eq', 'fun': lambda w, t=target: (w @ mean_returns) - t}
        ]
        
        res = sco.minimize(
            fun=fast_volatility, 
            x0=init_weights, 
            args=(cov_matrix,), 
            method='SLSQP', 
            bounds=bounds, 
            constraints=frontier_constraints
        )
        
        if res['success']:
            frontier_vols.append(res['fun'])
            valid_returns.append(target)
            transition_weights_list.append(res['x'])

    # Convertimos resultados a arrays nativos para cálculos masivos finales
    frontier_vols_arr = np.array(frontier_vols)
    valid_returns_arr = np.array(valid_returns)
    
    efficient_frontier_points = pd.DataFrame({
        'Return': valid_returns_arr,
        'Risk': frontier_vols_arr,
        'SharpeRatio': (valid_returns_arr - riskfreerate) / frontier_vols_arr
    })

    transition_map_points = pd.DataFrame(
        transition_weights_list, 
        columns=pfolio_assets, 
        index=frontier_vols_arr
    )
    
    # Riesgo de cola ejecutado una sola vez al final
    opt_tailriskmetrics = portfolio_valueatrisk(
        optimal_weights, dailyreturn, anualperiod, confidencelevel, numsimulations
    )
    
    optimal_portfolio = {
        'Return': opt_ret, 
        'Risk': opt_vol, 
        'SharpeRatio': opt_sh, 
        **{k: opt_tailriskmetrics[k] for k in [
            'VaR_Hist', 'CVaR_Hist', 'VaR_Param', 'CVaR_Param', 'VaR_MC', 'CVaR_MC'
        ]}
    }
    
    return optimal_weights, optimal_portfolio, efficient_frontier_points, transition_map_points 

def negate_sharpe(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, riskfreerate: float) -> float:
    """Calcula el Sharpe negativo reduciendo las operaciones de raíz cuadrada y divisiones complejas."""
    p_ret = weights @ mean_returns
    p_vol = np.sqrt(weights @ cov_matrix @ weights)
    return -(p_ret - riskfreerate) / p_vol

def fast_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Devuelve la volatilidad usando el operador @, optimizado para operaciones BLAS a nivel de CPU."""
    return np.sqrt(weights @ cov_matrix @ weights)