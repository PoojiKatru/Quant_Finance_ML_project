"""
features.py — Technical indicator & feature engineering pipeline.
Generates alpha signals used as model inputs.
"""

import numpy as np
import pandas as pd



def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def rate_of_change(series: pd.Series, window: int = 10) -> pd.Series:
    return series.pct_change(window)




def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / sma
    percent_b = (series - lower) / (upper - lower)
    return upper, sma, lower, bandwidth, percent_b


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def realized_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)




def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (direction * volume).fillna(0).cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume / volume.rolling(window).mean()




def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = atr(high, low, close, window)
    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(window).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(window).mean() / tr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(span=window, adjust=False).mean()




def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build full feature matrix from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: Open, High, Low, Close, Volume

    Returns
    -------
    pd.DataFrame
        Feature matrix with all engineered signals
    """
    feat = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    returns = close.pct_change()

    # --- Returns at multiple horizons ---
    for h in [1, 3, 5, 10, 21]:
        feat[f"ret_{h}d"] = close.pct_change(h)

    # --- Moving averages (ratio to close) ---
    for w in [5, 10, 20, 50, 200]:
        feat[f"sma_{w}_ratio"] = close / close.rolling(w).mean() - 1
        feat[f"ema_{w}_ratio"] = close / close.ewm(span=w, adjust=False).mean() - 1

    # --- Momentum ---
    feat["rsi_14"] = rsi(close, 14)
    feat["rsi_28"] = rsi(close, 28)
    macd_line, signal_line, hist = macd(close)
    feat["macd"] = macd_line
    feat["macd_signal"] = signal_line
    feat["macd_hist"] = hist
    feat["roc_10"] = rate_of_change(close, 10)
    feat["roc_21"] = rate_of_change(close, 21)

    # --- Volatility ---
    _, _, _, feat["bb_bandwidth"], feat["bb_pct_b"] = bollinger_bands(close)
    feat["atr_14"] = atr(high, low, close, 14)
    feat["atr_pct"] = feat["atr_14"] / close
    feat["realized_vol_21"] = realized_volatility(returns, 21)
    feat["realized_vol_63"] = realized_volatility(returns, 63)

    # --- Volume ---
    feat["obv"] = obv(close, volume)
    feat["obv_sma_ratio"] = feat["obv"] / feat["obv"].rolling(20).mean() - 1
    feat["volume_ratio_20"] = volume_ratio(volume, 20)
    feat["vwap_ratio"] = close / vwap(high, low, close, volume) - 1

    # --- Trend ---
    feat["adx_14"] = adx(high, low, close, 14)

    # --- Price structure ---
    feat["high_low_range"] = (high - low) / close
    feat["close_position"] = (close - low) / (high - low).replace(0, np.nan)
    feat["gap"] = (df["Open"] - close.shift()) / close.shift()

    # --- Lag features (autoregressive) ---
    for lag in [1, 2, 3, 5]:
        feat[f"ret_lag_{lag}"] = returns.shift(lag)
        feat[f"vol_lag_{lag}"] = feat["realized_vol_21"].shift(lag)

    # --- Forward return (target) ---
    feat["target_5d"] = close.pct_change(5).shift(-5)
    feat["target_direction"] = (feat["target_5d"] > 0).astype(int)

    return feat.dropna()
