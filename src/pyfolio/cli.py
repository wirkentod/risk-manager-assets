"""Simple CLI to compute and optionally save/plot correlations."""
import argparse, time
from src.pyfolio import load_data, compute_correlation, save_corr, compute_portfolio_metrics, compute_montecarlo_simulation, plot_heatmap, plot_montecarlo_simulation, RISK_FREE_RATE, ANUAL_PERIOD, ASSETS, WEIGHTS, NUM_SIMULATIONS

def build_parser():
    p = argparse.ArgumentParser(description="Compute asset correlation matrix from CSV")
    p.add_argument("input", help="path to input CSV file")
    p.add_argument("--dailyreturn", default="log", choices=("simple", "log"), help="daily return method")
    p.add_argument("--term", default="3M", choices=("1W", "1M", "2M", "3M", "1A"), help="term scenario to compute")
    p.add_argument("--method", default="pearson", choices=("pearson", "spearman"), help="correlation method")
    p.add_argument("--out", help="path to save CSV of correlation matrix")
    p.add_argument("--plot", help="path to save heatmap image (PNG) or omit to show")
    p.add_argument("--plotmontecarlo", help="path to save Monte Carlo simulation plot (PNG) or omit to show")
    return p


def main(argv=None):
    print(f"Start:")
    ini_load = time.perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)
    df = load_data(args.input)
    fin_load =time.perf_counter()
    print(f"Data loaded in: {fin_load - ini_load:.4f} seconds.")
    corr = compute_correlation(df, args.term, args.dailyreturn,  method=args.method)
    
    ini_risk = time.perf_counter()
    pfolio_assets, pfolio_weights = ASSETS, WEIGHTS
    returnP, riskP, sharpeP = compute_portfolio_metrics(df, args.term, args.dailyreturn, ANUAL_PERIOD, pfolio_assets, pfolio_weights)
    fin_risk = time.perf_counter()
    print(f"Portfolio metrics computed in: {fin_risk - ini_risk:.4f} seconds.")
    print(f"Portfolio Annualized Return: {returnP}")
    print(f"Portfolio Annualized Risk: {riskP}")
    print(f"Portfolio Annualized Sharpe Ratio: {sharpeP}")

    ini_montecarlo = time.perf_counter()
    optimal_weights, optimal_portfolio, simulated_portfolios = compute_montecarlo_simulation(df, args.term, args.dailyreturn, ANUAL_PERIOD, RISK_FREE_RATE, pfolio_assets, NUM_SIMULATIONS)
    fin_montecarlo = time.perf_counter()
    print(f"Monte Carlo simulation computed in: {fin_montecarlo - ini_montecarlo:.4f} seconds.")

    if args.out:
        save_corr(corr, args.out)
    if args.plot:
        plot_heatmap(corr, args.plot)
        ini_plotmontecarlo = time.perf_counter()
        plot_montecarlo_simulation(optimal_weights, optimal_portfolio, simulated_portfolios, pfolio_assets, returnP, riskP, sharpeP, args.plotmontecarlo)
        fin_plotmontecarlo = time.perf_counter()
        print(f"Monte Carlo simulation plot computed in: {fin_plotmontecarlo - ini_plotmontecarlo:.4f} seconds.")
    else:
        print(corr.to_string())
    

if __name__ == "__main__":
    main()
