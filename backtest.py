"""
backtest.py — Event-driven backtesting engine with realistic transaction costs.
Supports long-only, long-short, and signal-threshold strategies.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────
# Performance Metrics
# ─────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.04) -> float:
    excess = returns - risk_free / 252
    if excess.std() == 0:
        return 0.0
    return np.sqrt(252) * excess.mean() / excess.std()


def sortino_ratio(returns: pd.Series, risk_free: float = 0.04) -> float:
    excess = returns - risk_free / 252
    downside = excess[excess < 0].std()
    if downside == 0:
        return 0.0
    return np.sqrt(252) * excess.mean() / downside


def max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
    ann_return = (1 + returns.mean()) ** 252 - 1
    mdd = abs(max_drawdown(equity_curve))
    return ann_return / mdd if mdd != 0 else 0.0


def information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    active_returns = strategy_returns - benchmark_returns
    te = active_returns.std() * np.sqrt(252)
    return (active_returns.mean() * 252) / te if te != 0 else 0.0


def win_rate(returns: pd.Series) -> float:
    return (returns > 0).sum() / len(returns)


def profit_factor(returns: pd.Series) -> float:
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    return gross_profit / gross_loss if gross_loss != 0 else np.inf


# ─────────────────────────────────────────────
# Trade Record
# ─────────────────────────────────────────────

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    direction: int  # 1 = long, -1 = short
    size: float
    pnl: float = 0.0
    commission: float = 0.0


# ─────────────────────────────────────────────
# Backtester
# ─────────────────────────────────────────────

class Backtester:
    """
    Signal-based backtester with realistic cost modeling.

    Parameters
    ----------
    initial_capital : float
        Starting portfolio value
    commission_bps : float
        One-way commission in basis points (default: 5bps)
    slippage_bps : float
        One-way slippage in basis points (default: 3bps)
    max_position_pct : float
        Max allocation per position as % of portfolio (default: 100%)
    long_threshold : float
        Signal probability above which to go long
    short_threshold : float
        Signal probability below which to go short (for long-short mode)
    mode : str
        'long_only' or 'long_short'
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission_bps: float = 5.0,
        slippage_bps: float = 3.0,
        max_position_pct: float = 1.0,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
        mode: str = "long_only",
    ):
        self.initial_capital = initial_capital
        self.commission = commission_bps / 10_000
        self.slippage = slippage_bps / 10_000
        self.max_position_pct = max_position_pct
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.mode = mode

    def _transaction_cost(self, price: float, size: float) -> float:
        return price * size * (self.commission + self.slippage)

    def run(
        self,
        prices: pd.Series,
        signals: pd.Series,
        benchmark: Optional[pd.Series] = None
    ) -> dict:
        """
        Execute backtest over price + signal series.

        Parameters
        ----------
        prices : pd.Series
            Daily closing prices
        signals : pd.Series
            Model probability output [0, 1], same index as prices
        benchmark : pd.Series, optional
            Benchmark prices for comparison (e.g. buy-and-hold)

        Returns
        -------
        dict with equity curve, returns, metrics, and trade log
        """
        capital = self.initial_capital
        position = 0.0          # shares held
        direction = 0            # 1 long, -1 short, 0 flat
        equity_curve = []
        daily_returns = []
        trades: List[Trade] = []
        entry_price = None
        entry_date = None

        dates = prices.index
        prices_arr = prices.values
        signals_arr = signals.reindex(prices.index).values

        for i, (date, price, signal) in enumerate(zip(dates, prices_arr, signals_arr)):
            if np.isnan(signal):
                equity_curve.append(capital + position * price * direction)
                if i > 0:
                    daily_returns.append(0.0)
                continue

            # Determine target direction
            if signal >= self.long_threshold:
                target = 1
            elif self.mode == "long_short" and signal <= self.short_threshold:
                target = -1
            else:
                target = 0

            # Rebalance if direction changes
            if target != direction:
                # Exit current position
                if direction != 0 and position > 0:
                    cost = self._transaction_cost(price, position)
                    pnl = direction * position * (price - entry_price) - cost
                    trades.append(Trade(
                        entry_date=entry_date,
                        exit_date=date,
                        entry_price=entry_price,
                        exit_price=price,
                        direction=direction,
                        size=position,
                        pnl=pnl,
                        commission=cost
                    ))
                    capital += direction * position * price - cost
                    position = 0.0
                    direction = 0

                # Enter new position
                if target != 0:
                    alloc = capital * self.max_position_pct
                    position = alloc / price
                    cost = self._transaction_cost(price, position)
                    capital -= alloc + cost
                    direction = target
                    entry_price = price
                    entry_date = date

            portfolio_value = capital + (position * price * direction if direction != 0 else 0)
            equity_curve.append(portfolio_value)
            if i > 0:
                prev = equity_curve[-2]
                daily_returns.append((portfolio_value - prev) / prev if prev != 0 else 0)

        equity = pd.Series(equity_curve, index=dates)
        ret = pd.Series(daily_returns, index=dates[1:])

        # Benchmark (buy-and-hold)
        bh_returns = None
        if benchmark is not None:
            bh_returns = benchmark.pct_change().dropna().reindex(ret.index).fillna(0)

        # Compute metrics
        ann_return = (equity.iloc[-1] / self.initial_capital) ** (252 / len(equity)) - 1
        metrics = {
            "Total Return (%)": round((equity.iloc[-1] / self.initial_capital - 1) * 100, 2),
            "Annualized Return (%)": round(ann_return * 100, 2),
            "Sharpe Ratio": round(sharpe_ratio(ret), 3),
            "Sortino Ratio": round(sortino_ratio(ret), 3),
            "Max Drawdown (%)": round(max_drawdown(equity) * 100, 2),
            "Calmar Ratio": round(calmar_ratio(ret, equity), 3),
            "Win Rate (%)": round(win_rate(ret) * 100, 2),
            "Profit Factor": round(profit_factor(ret), 3),
            "Total Trades": len(trades),
            "Avg Trade PnL ($)": round(np.mean([t.pnl for t in trades]) if trades else 0, 2),
        }

        if bh_returns is not None:
            metrics["Information Ratio"] = round(information_ratio(ret, bh_returns), 3)
            bh_equity = (1 + bh_returns).cumprod() * self.initial_capital
            metrics["Benchmark Return (%)"] = round((bh_equity.iloc[-1] / self.initial_capital - 1) * 100, 2)

        return {
            "equity_curve": equity,
            "returns": ret,
            "metrics": metrics,
            "trades": trades,
            "benchmark_equity": bh_equity if bh_returns is not None else None
        }

    def print_summary(self, results: dict):
        print("\n" + "=" * 50)
        print("  BACKTEST RESULTS")
        print("=" * 50)
        for k, v in results["metrics"].items():
            print(f"  {k:<30} {v}")
        print("=" * 50)
