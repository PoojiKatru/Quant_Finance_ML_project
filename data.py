"""
data.py — Market data ingestion via yfinance with caching support.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


CACHE_DIR = Path("/tmp/quant_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance with local CSV caching.

    Parameters
    ----------
    ticker : str
        e.g. 'SPY', 'AAPL', 'QQQ'
    start, end : str
        Date strings 'YYYY-MM-DD'
    use_cache : bool
        If True, load from local cache if available

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
    """
    cache_path = CACHE_DIR / f"{ticker}_{start}_{end}.csv"

    if use_cache and cache_path.exists():
        print(f"[Data] Loading {ticker} from cache...")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    try:
        import yfinance as yf
        print(f"[Data] Fetching {ticker} from Yahoo Finance ({start} → {end})...")
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.to_csv(cache_path)
        print(f"[Data] Cached to {cache_path}")
        return df

    except ImportError:
        print("[Data] yfinance not installed. Generating synthetic data for demo...")
        return _generate_synthetic_data(ticker, start, end)


def _generate_synthetic_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for demo purposes."""
    np.random.seed(42)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Geometric Brownian Motion
    mu = 0.08 / 252
    sigma = 0.18 / np.sqrt(252)
    returns = np.random.normal(mu, sigma, n)

    # Add momentum regime
    regime = np.sin(np.linspace(0, 6 * np.pi, n)) * 0.002
    returns += regime

    prices = 400 * np.exp(np.cumsum(returns))

    daily_range = np.abs(np.random.normal(0, sigma * 2, n)) * prices
    opens = prices * (1 + np.random.normal(0, 0.003, n))
    highs = np.maximum(prices, opens) + daily_range * 0.5
    lows = np.minimum(prices, opens) - daily_range * 0.5
    volume = np.random.lognormal(np.log(5e7), 0.5, n).astype(int)

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": prices,
        "Volume": volume
    }, index=dates)

    print(f"[Data] Generated {len(df)} days of synthetic data for {ticker}")
    return df


def train_test_split_time(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> tuple:
    """
    Time-series aware train/val/test split (no shuffling).

    Returns
    -------
    (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        df.iloc[:train_end],
        df.iloc[train_end:val_end],
        df.iloc[val_end:]
    )


def add_market_regime(df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Label market regime: bull (1), bear (-1), neutral (0)."""
    rolling_ret = df["Close"].pct_change(window)
    vol = df["Close"].pct_change().rolling(window).std() * np.sqrt(252)

    conditions = [
        (rolling_ret > 0.05) & (vol < 0.25),
        (rolling_ret < -0.05) | (vol > 0.35),
    ]
    df["regime"] = np.select(conditions, [1, -1], default=0)
    return df
