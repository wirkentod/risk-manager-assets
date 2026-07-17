"""Simple CLI to compute and optionally save/plot correlations."""
import argparse, time
from src.pyfolio import(
    load_data, 
    compute_daily_return,
    compute_correlation, 
    compute_covariance, 
    save_corr, 
    compute_risk_descomposition, 
    compute_pca, 
    compute_assets_metrics, 
    compute_portfolio_metrics, 
    compute_efficient_frontier, 
    plot_risk_descomposition, 
    plot_heatmap,
    plot_portfolio_frontier, 
    plot_transition_map, 
    plot_portfolio_pca,  
    plot_assets_metrics, 
    plot_efficient_frontier_metrics, 
    RISK_FREE_RATE, 
    ANUAL_PERIOD, 
    ASSETS, 
    WEIGHTS,
    CONFIDENCE_LEVEL,
    NUM_SIMULATIONS
)

def build_parser():
    p = argparse.ArgumentParser(description="Compute asset correlation matrix from CSV")
    p.add_argument("input", help="path to input CSV file")
    p.add_argument("--dailychange", default="log", choices=("simple", "log"), help="daily return change method")
    p.add_argument("--term", default="3M", choices=("1W", "1M", "2M", "3M", "6M", "1A", "2A"), help="term scenario to compute")
    p.add_argument("--method", default="pearson", choices=("pearson", "spearman"), help="correlation method")
    p.add_argument("--out", help="path to save CSV of correlation matrix")
    p.add_argument("--plot", help="path to save heatmap image (PNG) or omit to show")
    p.add_argument("--plotfolio", help="path to save portfolio frontier plot (PNG) or omit to show")
    return p

def main(argv=None):
    print(f"Start:")
    ini_load = time.perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)
    datafolio = load_data(args.input, ASSETS)
    daily_return = compute_daily_return(datafolio, args.term, args.dailychange).dropna()
    print(f"\033[34mData loaded in: {time.perf_counter() - ini_load:.4f} seconds.\033[0m")
    corrfolio = compute_correlation(daily_return, method=args.method)
    covfolio = compute_covariance(daily_return)
    covfolioanual = covfolio * ANUAL_PERIOD
    ini_risk = time.perf_counter()
    pfolio_assets, pfolio_weights = ASSETS, WEIGHTS
    returnP, riskP, sharpeP, varP, cvarP = compute_portfolio_metrics(daily_return, ANUAL_PERIOD, RISK_FREE_RATE, pfolio_assets, pfolio_weights, CONFIDENCE_LEVEL)
    print(f"Portfolio Annualized Return: {returnP*100:.2f}%")
    print(f"Portfolio Annualized Risk: {riskP*100:.2f}%")
    print(f"Portfolio Annualized Sharpe Ratio: {sharpeP:.2f}x")
    print(f"Portfolio Annualized VaR: {varP*100:.2f}%")
    print(f"Portfolio Annualized CVaR: {cvarP*100:.2f}%")
    # Compute metrics assets by portfolio
    assets_metrics = compute_assets_metrics(daily_return, ANUAL_PERIOD, RISK_FREE_RATE, WEIGHTS, CONFIDENCE_LEVEL, NUM_SIMULATIONS)
    plot_assets_metrics(assets_metrics)
    # Compute pca metrics by portfolio
    eigenvaluesfolio, eigenvectorsfolio = compute_pca(daily_return, ANUAL_PERIOD, ASSETS)
    riskdescomposition = compute_risk_descomposition(covfolioanual, pfolio_assets, pfolio_weights, riskP)
    plot_risk_descomposition(riskdescomposition)
    print(f"\033[34mPortfolio metrics computed in: {time.perf_counter() - ini_risk:.4f} seconds.\033[0m")
    ini_effrontier = time.perf_counter()
    optimal_weights, optimal_portfolio, efficient_frontier_points, transition_map_points = compute_efficient_frontier(daily_return, ANUAL_PERIOD, RISK_FREE_RATE, pfolio_assets, CONFIDENCE_LEVEL)
    optimalriskdescomposition = compute_risk_descomposition(covfolioanual, pfolio_assets, optimal_weights, optimal_portfolio['Risk'])
    plot_efficient_frontier_metrics(optimal_weights, optimal_portfolio, pfolio_assets)
    plot_risk_descomposition(optimalriskdescomposition)
    print(f"\033[34mEfficient frontier computed in: {time.perf_counter() - ini_effrontier:.4f} seconds.\033[0m")
        
    # Plot results
    if args.out:
        save_corr(corrfolio, args.out)
    if args.plot:
        plot_portfolio_pca(eigenvaluesfolio, eigenvectorsfolio, corrfolio)
        plot_heatmap(corrfolio, args.plot)
        ini_plotportfolio = time.perf_counter()
        name_plot_portfolio = args.plotfolio + args.term + "_" + args.dailychange + "_frontier.png" if args.plotfolio else None
        plot_portfolio_frontier(optimal_weights, optimal_portfolio, efficient_frontier_points, pfolio_assets, assets_metrics, returnP, riskP, sharpeP, name_plot_portfolio)
        print(f"\033[34mEfficient frontier plot computed in: {time.perf_counter() - ini_plotportfolio:.4f} seconds.\033[0m")
        name_plot_transitionmap = args.plotfolio + args.term + "_" + args.dailychange + "_transition_map.png" if args.plotfolio else None
        ini_plottransition = time.perf_counter()
        plot_transition_map(transition_map_points, name_plot_transitionmap)
        print(f"\033[34mTransition map plot computed in: {time.perf_counter() - ini_plottransition:.4f} seconds.\033[0m")
        
    else:
        print(corrfolio.to_string())
    
if __name__ == "__main__":
    main()