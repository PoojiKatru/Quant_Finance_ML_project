"""
pipeline.py — End-to-end orchestration: data → features → train → backtest → report.
"""

import numpy as np
import pandas as pd

from data import fetch_ohlcv, train_test_split_time
from features import build_features
from models import get_model
from backtest import Backtester


def run_pipeline(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str = "2024-01-01",
    model_type: str = "ensemble",
    initial_capital: float = 100_000,
    plot: bool = False,
):
    print(f"\n{'='*60}")
    print(f"  QuantML Pipeline  |  {ticker}  |  {model_type.upper()}")
    print(f"  Period: {start} → {end}")
    print(f"{'='*60}")

    # ── 1. Data ──────────────────────────────────────────────
    print("\n[1/5] Fetching market data...")
    df = fetch_ohlcv(ticker, start, end)

    # ── 2. Feature Engineering ───────────────────────────────
    print("[2/5] Engineering features...")
    features_df = build_features(df)

    target_col = "target_direction"
    feature_cols = [c for c in features_df.columns if c not in ("target_5d", "target_direction")]

    X = features_df[feature_cols].values
    y = features_df[target_col].values

    # Time-series split
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"   Train: {train_end} | Val: {val_end - train_end} | Test: {n - val_end} samples")

    # ── 3. Model Training ────────────────────────────────────
    print(f"[3/5] Training {model_type} model...")
    model = get_model(model_type)

    if model_type == "ensemble":
        model.fit(X_train, y_train, X_val, y_val)
    else:
        model.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))

    # ── 4. Evaluation ─────────────────────────────────────────
    print("[4/5] Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    print(f"   Test Accuracy : {metrics['accuracy']:.4f}")
    print(f"   Test ROC-AUC  : {metrics['roc_auc']:.4f}")

    # ── 5. Backtest ───────────────────────────────────────────
    print("[5/5] Running backtest...")

    # Generate probability signals over test period
    test_index = features_df.index[val_end:]
    proba = model.predict_proba(X_test)

    # Align signal and prices
    test_prices = df["Close"].reindex(test_index)
    signals = pd.Series(proba[:len(test_index)], index=test_index)

    bt = Backtester(
        initial_capital=initial_capital,
        long_threshold=0.55,
        short_threshold=0.45,
        mode="long_only",
    )
    results = bt.run(test_prices, signals, benchmark=test_prices)
    bt.print_summary(results)

    # ── Feature Importance ────────────────────────────────────
    if hasattr(model, "feature_importance"):
        fi = model.feature_importance(feature_cols)
        if len(fi) > 0:
            print("\nTop 10 Features:")
            for feat, imp in fi.head(10).items():
                bar = "█" * int(imp * 200)
                print(f"  {feat:<30} {bar} {imp:.4f}")

    # ── Optional Plot ─────────────────────────────────────────
    if plot:
        _plot_results(results, ticker, model_type)

    return results


def _plot_results(results: dict, ticker: str, model_type: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        fig.suptitle(f"QuantML — {ticker} | {model_type.upper()}", fontsize=14, fontweight="bold")

        equity = results["equity_curve"]
        returns = results["returns"]
        benchmark = results.get("benchmark_equity")

        # Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(equity, label="Strategy", color="#2196F3", linewidth=2)
        if benchmark is not None:
            ax1.plot(benchmark, label="Buy & Hold", color="#FF9800", linewidth=1.5, alpha=0.8)
        ax1.set_title("Equity Curve")
        ax1.legend()
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.grid(alpha=0.3)

        # Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        drawdown = (equity - equity.cummax()) / equity.cummax() * 100
        ax2.fill_between(drawdown.index, drawdown, 0, color="#F44336", alpha=0.6)
        ax2.set_title("Drawdown (%)")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(alpha=0.3)

        # Returns distribution
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.hist(returns * 100, bins=50, color="#4CAF50", edgecolor="white", alpha=0.8)
        ax3.axvline(0, color="red", linestyle="--")
        ax3.set_title("Daily Returns Distribution")
        ax3.set_xlabel("Return (%)")
        ax3.grid(alpha=0.3)

        # Rolling Sharpe
        ax4 = fig.add_subplot(gs[2, 1])
        rolling_sharpe = returns.rolling(63).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() != 0 else 0
        )
        ax4.plot(rolling_sharpe, color="#9C27B0", linewidth=1.5)
        ax4.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax4.axhline(1, color="green", linestyle="--", alpha=0.5)
        ax4.set_title("Rolling Sharpe Ratio (63d)")
        ax4.grid(alpha=0.3)

        plt.tight_layout()

        out_path = f"results/{ticker}_{model_type}_backtest.png"
        import os; os.makedirs("results", exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\n[Plot] Saved to {out_path}")
        plt.show()

    except ImportError:
        print("[Plot] matplotlib not installed. Skipping plot.")
