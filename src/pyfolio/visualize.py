"""Visualization helpers for correlation matrices."""
from adjustText import adjust_text
from pathlib import Path
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_heatmap(corr, out_path=None, figsize=(10, 8), cmap="vlag"):
    """Plot or save a correlation heatmap.

    If `out_path` is None the plot will be shown interactively.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0)
    plt.tight_layout()
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p)
        plt.close()
    else:
        plt.show()

def plot_portfolio_frontier(optimal_weights, optimal_portfolio, simulated_portfolios, pfolio_assets, assets_metrics, returnP, riskP, sharpeP, out_path=None, figsize=(14, 8)):
    """
    Plots the Monte Carlo simulation ensuring all assets, legend, and info_text are perfectly aligned and visible.
    """
    # Prepare the figure and the main axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set percentage format with one decimal place for both axes
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.1f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.1f}%"))
    
    # Plot the simulation (Scatter plot of simulated portfolios)
    sc = ax.scatter(
        simulated_portfolios['Risk'], 
        simulated_portfolios['Return'], 
        c=simulated_portfolios['SharpeRatio'], 
        cmap='YlGnBu', 
        alpha=0.4,       
        edgecolors='none',
        zorder=1
    )
    
    # Plot specific portfolios and individual assets
    ax.scatter(riskP, returnP, color='red', marker='p', s=150, label='Current Portfolio', zorder=4)
    ax.scatter(optimal_portfolio['Risk'], optimal_portfolio['Return'], color='green', marker='o', s=150, label='Optimal Portfolio', zorder=5)
    ax.scatter(assets_metrics['Risk'], assets_metrics['Return'], color='black', marker='h', s=120, label='Individual Assets', zorder=3)
    
    # Annotate asset tickers for each individual asset
    # Optional: If adjustText library is installed, use it to automatically prevent overlaps.
    try:
        texts = []
        for ticker, asset in assets_metrics.iterrows():
            texts.append(ax.text(asset['Risk'], asset['Return'], ticker, fontsize=9, color='black', weight='bold', zorder=6, ha='center', va='bottom'))
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))
    except ImportError:
        # Fallback if adjustText is not available
        for ticker, asset in assets_metrics.iterrows():
            ax.annotate(
                ticker, 
                (asset['Risk'], asset['Return']), 
                textcoords="offset points", 
                xytext=(0, 8), 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                color='black', 
                weight='bold',
                zorder=6
            )
    
    # Set internal padding for the plot area
    ax.set_xmargin(0.12)
    ax.set_ymargin(0.12)
    
    # Titles and grid configuration
    ax.set_xlabel('Annualized Risk (Volatility)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Annualized Return', fontsize=11, fontweight='bold')
    ax.set_title('Efficient Frontier & Portfolio Optimization', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Generate the structure for the lateral information text box
    weight_lines = [f"  {asset}: {weight*100:.2f}%" for asset, weight in zip(pfolio_assets, optimal_weights)]
    
    info_text = (
        f"OPTIMAL WEIGHTS:\n" + "\n".join(weight_lines) + "\n\n"
        f"OPTIMIZED PORTFOLIO:\n"
        f"Return: {optimal_portfolio['Return'] * 100:.2f}%\n"
        f"Risk: {optimal_portfolio['Risk'] * 100:.2f}%\n"
        f"Sharpe Ratio: {optimal_portfolio['SharpeRatio']:.2f}\n\n"
        f"CURRENT PORTFOLIO:\n"
        f"Return: {returnP * 100:.2f}%\n"
        f"Risk: {riskP * 100:.2f}%\n"
        f"Sharpe Ratio: {sharpeP:.2f}"
    )
    
    # Adjust figure subplots explicitly to prevent random shifts
    fig.subplots_adjust(left=0.08, right=0.74, top=0.90, bottom=0.18)

    # Add the color bar attached to the main plot
    cbar = fig.colorbar(sc, ax=ax, pad=0.03)
    cbar.set_label('Sharpe Ratio', fontweight='bold', fontsize=11)

    # Create an invisible axis to dock the text box perfectly next to the color bar
    ax_text = fig.add_axes([0.81, 0.22, 0.16, 0.58])
    ax_text.axis('off')  # Hide borders and axes lines for the text container
    
    # Render the text box inside the newly anchored invisible container
    ax_text.text(
        0.0, 0.5, info_text, fontsize=9.5, fontfamily='monospace',
        verticalalignment='center', ha='left',
        bbox=dict(boxstyle='round,pad=0.6', edgecolor='#A0A0A0', facecolor='#FAFAFA', alpha=1.0)
    )

    # Place the markers legend horizontally at the bottom center
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15), 
        ncol=3, 
        framealpha=0.9,
        fontsize=10
    )

    # Save or display the figure ensuring tight constraints
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_transition_map(transition_map_df, out_path=None):
    """
    Plots a stacked area chart representing asset allocation changes 
    with dynamic text labels placed directly inside each color band.
    """
    plt.figure(figsize=(12, 7))
    
    x_risk = transition_map_df.index.values
    assets = transition_map_df.columns
    y_allocations = [transition_map_df[asset].values for asset in assets]
    
    # 1. Draw the base stacked area plot
    # We save the baseline poly-collections to extract accurate colors if needed
    polys = plt.stackplot(x_risk, y_allocations, labels=assets, alpha=0.85)
    
    # Calculate cumulative sums to find absolute height boundaries for text placement
    # shape: (num_assets, num_points)
    cumulative_weights = np.cumsum(np.vstack(y_allocations), axis=0)
    
    # 2. Mathematically inject dynamic text labels inside the bands
    for i, asset in enumerate(assets):
        weights = transition_map_df[asset].values
        
        # Skip assets that never get allocated to avoid zero-division or errors
        if np.max(weights) < 0.05:  # Only label if asset exceeds 5% allocation at peak
            continue
            
        # Find the specific X index where this specific asset reaches its maximum weight
        max_idx = np.argmax(weights)
        x_pos = x_risk[max_idx]
        
        # Calculate the vertical midpoint inside its specific color band at that peak X coordinate
        lower_bound = cumulative_weights[i-1, max_idx] if i > 0 else 0.0
        upper_bound = cumulative_weights[i, max_idx]
        y_pos = lower_bound + (upper_bound - lower_bound) / 2.0
        
        # Add text label directly inside the plot canvas
        plt.text(
            x=x_pos, 
            y=y_pos, 
            s=asset, 
            color='black', 
            fontsize=10, 
            weight='bold',
            va='center', 
            ha='center',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')
        )

    # --- FORMATTING AESTHETICS ---
    plt.title('Efficient Frontier Transition Map', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Standard Deviation (Risk)', fontsize=11, labelpad=10)
    plt.ylabel('Allocation (Weight %)', fontsize=11, labelpad=10)
    
    # Convert scales to clean percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
    
    plt.xlim(x_risk.min(), x_risk.max())
    plt.ylim(0, 1.0)
    
    # Keeps a backup bottom legend for assets with too small an area to hold text
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=True)
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Save or display the figure ensuring tight constraints
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_risk_descomposition(riskdescomposition):
    sorted_metrics = riskdescomposition[['Weight','RiskDesc','RDW']].sort_values(by='RiskDesc',ascending=False)
    formatted_metrics = sorted_metrics.copy()
    formatted_metrics['Weight'] = formatted_metrics['Weight'].map('{:.2%}'.format)
    formatted_metrics['RiskDesc'] = formatted_metrics['RiskDesc'].map('{:.2%}'.format)
    formatted_metrics['RDW'] = formatted_metrics['RDW'].map('{:.2f}x'.format)
    print(f"\033[33mRisk Descomposition:\n{formatted_metrics}\033[0m")

def plot_assets_metrics(assets_metrics, confidencelevel: float):
    sorted_metrics = assets_metrics[['Weight', 'SharpeRatio', 'Return', 'Risk', 'VaRHist', 'VaRParam', 'VaRMC','CVaRHist','CVaRParam','CVaRMC']].sort_values(by='SharpeRatio', ascending=False)
    formatted_metrics = sorted_metrics.copy()
    for metric in assets_metrics:
        if "ratio" in metric.lower():
            formatted_metrics[metric] = formatted_metrics[metric].map('{:.2f}x'.format)
        else:
            formatted_metrics[metric] = formatted_metrics[metric].map('{:.2%}'.format)
    print(f"\033[35mAsset metrics (Sharpe Ratio Ordered) - Tail risk metrics at ({confidencelevel:.0%} Confidence):\n{formatted_metrics}\033[0m")

def plot_portfolio_metrics(portfolio: dict, portfolioname: str, confidencelevel: float):
    """Prints a clean, high-impact executive risk report 
    using a portfolio metrics dictionary.
    """
    # Convert the key names to lowercase to avoid case-sensitivity errors
    p = {k.lower(): v for k, v in portfolio.items()}
    print("=" * 80)
    print(f"{portfolioname.upper() + ' PORTFOLIO':^70}")
    print("=" * 80)
    
    # Performance and Risk
    print("PERFORMANCE & EFFICIENCY METRICS")
    print("-" * 80)
    print(f" Expected Annual Return:     {p.get('return', 0):>8.2%}")
    print(f" Annual Volatility (Risk):   {p.get('risk', 0):>8.2%}")
    print(f" Annualized Sharpe Ratio:    {p.get('sharperatio', 0):>8.2f}x")
    print()
    
    # Tail Risk
    print(f"ANNUALIZED TAIL RISK METRICS ({confidencelevel:.0%} Confidence)")
    print("-" * 80)
    print(f" {'METHODOLOGY':<29} {'VaR':<17} {'CVaR':<17}")
    print(" ────────────────────────────────────────────────────────────────────")
    print(f" Historical                  {p.get('var_hist', 0):<17.2%} {p.get('cvar_hist', 0):.2%}")
    print(f" Parametric (Gaussian)       {p.get('var_param', 0):<17.2%} {p.get('cvar_param', 0):.2%}")
    print(f" Monte Carlo (Multivariate)  {p.get('var_mc', 0):<17.2%} {p.get('cvar_mc', 0):.2%}")
    print("=" * 80)

def plot_portfolio_pca(eigenvalues, eigenvectors, corrfolio, pcamargin):
    # Identifiy correlations extremes in portfolio
    top_extremes = 4
    upper_corr = corrfolio.where(np.triu(np.ones(corrfolio.shape), k=1).astype(bool))
    highest_corr = upper_corr.stack().nlargest(top_extremes)
    lowest_corr = upper_corr.stack().nsmallest(top_extremes)
    print("\nTop Systemic Redundancies (Highest Correlation):")
    for (a1, a2), val in highest_corr.items():
        print(f"  • {a1} ↔ {a2}: {val:.2f} (Potential concentration risk)")
        
    print("\nTop Diversification Drivers (Lowest/Negative Correlation):")
    for (a1, a2), val in lowest_corr.items():
        print(f"  • {a1} ↔ {a2}: {val:.2f} (Effective tail-risk offset)")   

    print("\nReporte de Estructura de Varianza Explicada")
    print("PRINCIPAL COMPONENT ANALYSIS (Latent Risk Factors Spectrum)")
    print("-" * 80)
    cumulative_var = 0
    for i, var in enumerate(eigenvalues):
        cumulative_var += var
        print(f"\033[32mPC{i+1} Eigenvalue Explanation: {var*100:.2f}% of total, accum: {cumulative_var*100:.2f}% portfolio variance.\033[0m")
    print("\n" + "-" * 80)
    print("FACTOR LOADINGS MATRIX (Asset Sensitivity to Latent Factors)")
    print("-" * 80)
    # Formatear la matriz de cargas para identificar rápidamente dominancias
    print(eigenvectors.round(4).to_string())

    eigenvalsum = np.cumsum(eigenvalues)
    num_pcx = np.sum(eigenvalsum <= pcamargin) #PCA_MARGIN: threshold cumulate eigenvalues (exm: 50%)
    eigenvectorsround = eigenvectors.round(4)
    for j in range(1, num_pcx + 1):
        print(f"\n{'-' * 80}")
        id_max = eigenvectorsround[f"PC{j}"].idxmax()
        id_min = eigenvectorsround[f"PC{j}"].idxmin()
        val_max = eigenvectorsround[f"PC{j}"][id_max]
        val_min = eigenvectorsround[f"PC{j}"][id_min]
        print(f"\033[31m• Factor PC{j}: ({id_max}, {val_max}) and ({id_min}, {val_min}).\033[0m")
        print(eigenvectorsround[f"PC{j}"].sort_values(ascending=False))