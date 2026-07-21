"""Simple CLI to compute and optionally save/plot correlations."""
import argparse, time
from src.pyfolio import(
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
    # weights
    compute_efficient_frontier, 
    compute_riskparity,
    # visualization functions
    plot_risk_descomposition, 
    plot_heatmap,
    plot_portfolio_frontier, 
    plot_transition_map, 
    plot_portfolio_pca,  
    plot_assets_metrics, 
    plot_portfolio_metrics, 
    # global parameters
    RISK_FREE_RATE, 
    ANUAL_PERIOD, 
    ASSETS, 
    WEIGHTS,
    CONFIDENCE_LEVEL,
    NUM_SIMULATIONS,
    PCA_MARGIN
)

def build_parser():
    p = argparse.ArgumentParser(description="Compute asset correlation matrix from CSV")
    p.add_argument("input", help="path to input CSV file")
    p.add_argument("--dailychange", default="log", choices=("simple", "log"), help="daily return change method")
    p.add_argument("--term", default="3M", choices=("1W", "1M", "2M", "3M", "6M", "1A", "2A", "3A", "5A", "6A"), help="term scenario to compute")
    p.add_argument("--offset", default="T0", choices=("T0","1W", "1M", "2M", "3M", "6M", "1A", "2A", "3A", "5A", "6A"), help="how many days prior to the present (T=0) your data window ends")
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
    daily_return = compute_daily_return(datafolio, args.offset, args.term, args.dailychange).dropna()
    print(f"\033[34mData loaded in: {time.perf_counter() - ini_load:.4f} seconds.\033[0m")
    corrfolio = compute_correlation(daily_return, method=args.method)
    covfolio = compute_covariance(daily_return)
    covfolioanual = covfolio * ANUAL_PERIOD
    ini_risk = time.perf_counter()
    pfolio_assets, pfolio_weights = ASSETS, WEIGHTS
    current_portfolio = compute_portfolio_metrics(daily_return, ANUAL_PERIOD, RISK_FREE_RATE, pfolio_assets, pfolio_weights, CONFIDENCE_LEVEL, NUM_SIMULATIONS)
    returnP = current_portfolio['Return']
    riskP = current_portfolio['Risk']
    sharpeP = current_portfolio['SharpeRatio']
    plot_portfolio_metrics(current_portfolio, 'Current', CONFIDENCE_LEVEL)
    # Compute metrics assets by portfolio
    assets_metrics = compute_assets_metrics(daily_return, ANUAL_PERIOD, RISK_FREE_RATE, WEIGHTS, CONFIDENCE_LEVEL, NUM_SIMULATIONS)
    plot_assets_metrics(assets_metrics, CONFIDENCE_LEVEL)
    # Compute pca metrics by portfolio
    eigenvaluesfolio, eigenvectorsfolio = compute_pca(daily_return, ANUAL_PERIOD, ASSETS)
    riskdescomposition = compute_risk_descomposition(covfolioanual, pfolio_assets, pfolio_weights, riskP)
    plot_risk_descomposition(riskdescomposition)
    print(f"\033[34mPortfolio metrics computed in: {time.perf_counter() - ini_risk:.4f} seconds.\033[0m")
    ini_effrontier = time.perf_counter()
    optimal_weights, optimal_portfolio, efficient_frontier_points, transition_map_points = compute_efficient_frontier(daily_return, ANUAL_PERIOD, RISK_FREE_RATE, pfolio_assets, CONFIDENCE_LEVEL, NUM_SIMULATIONS)
    optimalriskdescomposition = compute_risk_descomposition(covfolioanual, pfolio_assets, optimal_weights, optimal_portfolio['Risk'])
    plot_portfolio_metrics(optimal_portfolio, 'Optimal', CONFIDENCE_LEVEL)
    plot_risk_descomposition(optimalriskdescomposition)
    print(f"\033[34mEfficient frontier computed in: {time.perf_counter() - ini_effrontier:.4f} seconds.\033[0m")
    ini_riskparity = time.perf_counter()
    riskparityfolio_weights = compute_riskparity(daily_return)
    riskparity_portfolio = compute_portfolio_metrics(daily_return, ANUAL_PERIOD, RISK_FREE_RATE, pfolio_assets, riskparityfolio_weights, CONFIDENCE_LEVEL, NUM_SIMULATIONS)
    plot_portfolio_metrics(riskparity_portfolio, 'Risk Parity', CONFIDENCE_LEVEL)
    riskparitydescomposition = compute_risk_descomposition(covfolioanual, pfolio_assets, riskparityfolio_weights, riskparity_portfolio['Risk'])
    plot_risk_descomposition(riskparitydescomposition)
    print(f"\033[34mRisk Parity computed in: {time.perf_counter() - ini_riskparity:.4f} seconds.\033[0m")

    # Plot results
    if args.out:
        save_corr(corrfolio, args.out)
    if args.plot:
        plot_portfolio_pca(eigenvaluesfolio, eigenvectorsfolio, corrfolio, PCA_MARGIN)
        plot_heatmap(corrfolio, args.plot)
        ini_plotportfolio = time.perf_counter()
        name_plot_portfolio = args.plotfolio + "_offs" + args.offset + "_term" + args.term + "_" + args.dailychange + "_frontier.png" if args.plotfolio else None
        plot_portfolio_frontier(optimal_weights, optimal_portfolio, efficient_frontier_points, pfolio_assets, assets_metrics, returnP, riskP, sharpeP, name_plot_portfolio)
        print(f"\033[34mEfficient frontier plot computed in: {time.perf_counter() - ini_plotportfolio:.4f} seconds.\033[0m")
        name_plot_transitionmap = args.plotfolio + "_offs" + args.offset + "_term" + args.term + "_" + args.dailychange + "_transition_map.png" if args.plotfolio else None
        ini_plottransition = time.perf_counter()
        plot_transition_map(transition_map_points, name_plot_transitionmap)
        print(f"\033[34mTransition map plot computed in: {time.perf_counter() - ini_plottransition:.4f} seconds.\033[0m")
        
    else:
        print(corrfolio.to_string())
    
if __name__ == "__main__":
    main()