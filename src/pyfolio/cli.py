"""Simple CLI to compute and optionally save/plot correlations."""
import argparse, time
from src.pyfolio import(
    load_data, 
    compute_correlation, 
    save_corr, 
    compute_assets_metrics, 
    compute_portfolio_metrics, 
    compute_efficient_frontier, 
    plot_heatmap, 
    plot_portfolio_frontier, 
    plot_transition_map,    
    RISK_FREE_RATE, 
    ANUAL_PERIOD, 
    ASSETS, 
    WEIGHTS
)

def build_parser():
    p = argparse.ArgumentParser(description="Compute asset correlation matrix from CSV")
    p.add_argument("input", help="path to input CSV file")
    p.add_argument("--dailyreturn", default="log", choices=("simple", "log"), help="daily return method")
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
    df = load_data(args.input, ASSETS)
    fin_load =time.perf_counter()
    print(f"Data loaded in: {fin_load - ini_load:.4f} seconds.")
    corr = compute_correlation(df, args.term, args.dailyreturn,  method=args.method)
    # Compute metrics assets by portfolio
    assets_metrics = compute_assets_metrics(df, args.term, args.dailyreturn, ANUAL_PERIOD, RISK_FREE_RATE)

    ini_risk = time.perf_counter()
    pfolio_assets, pfolio_weights = ASSETS, WEIGHTS
    returnP, riskP, sharpeP = compute_portfolio_metrics(df, args.term, args.dailyreturn, ANUAL_PERIOD, RISK_FREE_RATE, pfolio_assets, pfolio_weights)
    fin_risk = time.perf_counter()
    print(f"Portfolio metrics computed in: {fin_risk - ini_risk:.4f} seconds.")
    print(f"Portfolio Annualized Return: {returnP}")
    print(f"Portfolio Annualized Risk: {riskP}")
    print(f"Portfolio Annualized Sharpe Ratio: {sharpeP}")

    ini_effrontier = time.perf_counter()
    optimal_weights, optimal_portfolio, efficient_frontier_points, transition_map_points = compute_efficient_frontier(df, args.term, args.dailyreturn, ANUAL_PERIOD, RISK_FREE_RATE, pfolio_assets)
    fin_effrontier = time.perf_counter()
    print(f"Efficient frontier computed in: {fin_effrontier - ini_effrontier:.4f} seconds.")
    
    # Plot results
    if args.out:
        save_corr(corr, args.out)
    if args.plot:
        plot_heatmap(corr, args.plot)
        ini_plotportfolio = time.perf_counter()
        name_plot_portfolio = args.plotfolio + args.term + "_frontier.png" if args.plotfolio else None
        plot_portfolio_frontier(optimal_weights, optimal_portfolio, efficient_frontier_points, pfolio_assets, assets_metrics, returnP, riskP, sharpeP, name_plot_portfolio)
        fin_plotportfolio = time.perf_counter()
        print(f"Efficient frontier plot computed in: {fin_plotportfolio - ini_plotportfolio:.4f} seconds.")
        name_plot_transitionmap = args.plotfolio + args.term + "_transition_map.png" if args.plotfolio else None
        ini_plottransition = time.perf_counter()
        plot_transition_map(transition_map_points, name_plot_transitionmap)
        fin_plottransition = time.perf_counter()
        print(f"Transition map plot computed in: {fin_plottransition - ini_plottransition:.4f} seconds.")
        
    else:
        print(corr.to_string())
    
if __name__ == "__main__":
    main()