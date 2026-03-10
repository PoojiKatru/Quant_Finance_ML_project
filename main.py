"""
QuantML: Equity Return Prediction & Backtesting Engine
Entry point for the full pipeline.
"""

import argparse
from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="QuantML Pipeline")
    parser.add_argument("--ticker", type=str, default="SPY", help="Stock ticker symbol")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--model", type=str, default="ensemble",
                        choices=["xgboost", "lstm", "random_forest", "ensemble"],
                        help="Model type to use")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Initial capital ($)")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    args = parser.parse_args()

    run_pipeline(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        model_type=args.model,
        initial_capital=args.initial_capital,
        plot=args.plot
    )


if __name__ == "__main__":
    main()
