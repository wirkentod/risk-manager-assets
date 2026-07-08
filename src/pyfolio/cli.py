"""Simple CLI to compute and optionally save/plot correlations."""
import argparse, time
from .core import load_data, compute_correlation, save_corr, compute_risk, compute_daily_return
from .visualize import plot_heatmap


def build_parser():
    p = argparse.ArgumentParser(description="Compute asset correlation matrix from CSV")
    p.add_argument("input", help="path to input CSV file")
    p.add_argument("--dailyreturn", default="log", choices=("simple", "log"), help="daily return method")
    p.add_argument("--term", default="3M", choices=("1W", "1M", "2M", "3M", "1A"), help="term scenario to compute")
    p.add_argument("--method", default="pearson", choices=("pearson", "spearman"), help="correlation method")
    p.add_argument("--out", help="path to save CSV of correlation matrix")
    p.add_argument("--plot", help="path to save heatmap image (PNG) or omit to show")
    return p


def main(argv=None):
    print("Comencemos:")
    ini_load = time.perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)
    df = load_data(args.input)
    fin_load =time.perf_counter()
    print(f"Data loaded in: {fin_load - ini_load:.4f} seconds.")
    corr = compute_correlation(df, args.term, args.dailyreturn,  method=args.method)
    
    ini_risk = time.perf_counter()
    risk = compute_risk(df, args.term, args.dailyreturn, 252)
    fin_risk = time.perf_counter()
    print(f"Risk computed in: {fin_risk - ini_risk:.4f} seconds.")
    
    if args.out:
        save_corr(corr, args.out)
    if args.plot:
        plot_heatmap(corr, args.plot)
    else:
        print(corr.to_string())
    

if __name__ == "__main__":
    main()
