"""Visualization helpers for correlation matrices."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


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
        from adjustText import adjust_text
        texts = []
        for ticker, asset in assets_metrics.iterrows():
            texts.append(ax.text(asset['Risk'], asset['Return'], ticker, fontsize=9, color='black', weight='bold', zorder=6))
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