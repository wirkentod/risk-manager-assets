"""Packet utilities for assets."""

# =====================================================================
# 1. CORE & VISUALIZATION (Function and Logic)
# =====================================================================
from .core import (
    # I/O files data
    load_data,
    save_corr,
    # process data
    compute_daily_return,
    compute_correlation,
    compute_covariance,
    # basic metrics
    compute_assets_metrics,
    compute_portfolio_metrics,
    # medium metrics
    compute_risk_descomposition, 
    compute_pca,
    compute_montecarlo_simulation,
    # weights
    compute_efficient_frontier,
    compute_riskparity
)
from .visualize import (
    # visualization functions
    plot_risk_descomposition, 
    plot_heatmap, 
    plot_portfolio_frontier,
    plot_transition_map,
    plot_portfolio_pca,
    plot_assets_metrics,
    plot_portfolio_metrics
)

# =====================================================================
# 2. CONFIGURATIONS & GLOBAL PARAMETERS (Loaded from JSON)
# =====================================================================
from .config.lector import (
    RISK_FREE_RATE, 
    ANUAL_PERIOD, 
    ASSETS, 
    WEIGHTS, 
    CONFIDENCE_LEVEL, 
    NUM_SIMULATIONS
)

# =====================================================================
# 3. API PUBLIC (Export control)
# =====================================================================
__all__ = [
    # I/O files data
    "load_data", 
    "save_corr",
    # process data
    "compute_daily_return", 
    "compute_correlation",
    "compute_covariance",
    # basic metrics
    "compute_assets_metrics",
    "compute_portfolio_metrics",
    # medium metrics
    "compute_risk_descomposition",
    "compute_pca",
    "compute_montecarlo_simulation",
    # weights
    "compute_efficient_frontier",
    "compute_riskparity"
    # visualization functions
    "plot_risk_descomposition", 
    "plot_heatmap",
    "plot_portfolio_frontier",
    "plot_transition_map",
    "plot_portfolio_pca",
    "plot_assets_metrics",
    "plot_portfolio_metrics",
    # global variables
    "RISK_FREE_RATE",
    "ANUAL_PERIOD",
    "ASSETS",
    "WEIGHTS",
    "CONFIDENCE_LEVEL",
    "NUM_SIMULATIONS",
]