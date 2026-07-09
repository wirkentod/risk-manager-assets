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

def plot_montecarlo_simulation(optimal_weights, optimal_portfolio, simulated_portfolios, pfolio_assets, assets_metrics, returnP, riskP, sharpeP, out_path=None, figsize=(12, 8)):
    """Plot or save a Monte Carlo simulation scatter plot.

    If `out_path` is None the plot will be shown interactively.
    """
    # Visualization
    plt.figure(figsize=figsize)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.1f}%'.format(x * 100)))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}%'.format(y * 100)))
    plt.scatter(simulated_portfolios['Risk'], simulated_portfolios['Return'], c=simulated_portfolios['SharpeRatio'], cmap='YlGnBu')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    plt.scatter(riskP, returnP, color='red', marker='p', s=100)  # Mark the Portfolio performance
    plt.scatter(optimal_portfolio['Risk'], optimal_portfolio['Return'], color='green', marker='o', s=100)  # Mark the optimal portfolio
    # Visualizate each asset in the portfolio
    plt.scatter(assets_metrics['Risk'], assets_metrics['Return'], color='black', marker='h', s=100)
    for ticker, asset in assets_metrics.iterrows():
        plt.annotate(ticker, (asset['Risk'], asset['Return']), textcoords="offset points", xytext=(0,8), ha='center', va='center', fontsize=8, color='black', weight='bold')
    
    # Displaying the weights of each item in the optimal portfolio
    weight_text = "Optimal Weights:\n" + '\n'.join([f"{asset}: {weight*100:.2f}%" for asset, weight in zip(pfolio_assets, optimal_weights)])
    plt.gcf().text(0.14, 0.86, weight_text, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'), ha='left')

    # Displaying Optimized Portfolio Performance
    optimal_text = f"Optimized Portfolio\nReturn: {optimal_portfolio['Return'] * 100:.2f}%\nRisk: {optimal_portfolio['Risk'] * 100:.2f}%\nSharpe Ratio: {optimal_portfolio['SharpeRatio']:.2f}"
    plt.gcf().text(0.14, (0.86 - 0.02 - (len(pfolio_assets) + 1) * 0.02), optimal_text, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='white'), ha='left')

    # Displaying Portfolio Performance
    market_text = f"(PEN Portfolio)\nReturn: {returnP * 100:.2f}%\nRisk: {riskP * 100:.2f}%\nSharpe Ratio: {sharpeP:.2f}"
    plt.gcf().text(0.14, (0.86 - 0.12 - (len(pfolio_assets) + 1) * 0.02), market_text, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white'), ha='left')

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p)
        plt.close()
    else:
        # Display Full Chart
        plt.show()