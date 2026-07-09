"""Packet utilities for assets."""

# =====================================================================
# 1. CORE & VISUALIZATION (Function and Logic)
# =====================================================================
from .core import load_data, compute_correlation, save_corr, compute_assets_metrics, compute_portfolio_metrics, compute_montecarlo_simulation
from .visualize import plot_heatmap, plot_montecarlo_simulation

# =====================================================================
# 2. CONFIGURATIONS & GLOBAL PARAMETERS (Loaded from JSON)
# =====================================================================
from .config.lector import RISK_FREE_RATE, ANUAL_PERIOD, ASSETS, WEIGHTS, NUM_SIMULATIONS

# =====================================================================
# 3. API PUBLIC (Export control)
# =====================================================================
__all__ = ["load_data", "compute_correlation", "save_corr", "compute_assets_metrics", "compute_portfolio_metrics","compute_montecarlo_simulation","plot_heatmap","plot_montecarlo_simulation", "RISK_FREE_RATE", "ANUAL_PERIOD", "ASSETS", "WEIGHTS", "NUM_SIMULATIONS"]
