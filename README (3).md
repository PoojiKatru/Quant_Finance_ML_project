# QuantML — Equity Return Prediction & Backtesting Engine

> A machine learning system for predicting short-term equity price direction and evaluating trading strategies through rigorous backtesting with realistic market frictions.

---

## Overview

QuantML is an end-to-end quantitative finance pipeline that combines **feature engineering**, **machine learning**, and **event-driven backtesting** to generate and evaluate trading signals on equity data. The system supports multiple model architectures — from gradient-boosted trees to LSTM networks with attention — and measures strategy performance against a buy-and-hold benchmark with transaction costs and slippage modeled explicitly.

```
Market Data → Feature Engineering → ML Model → Signal Generation → Backtest → Performance Report
```

---

## Features

### Data Pipeline
- Automatic OHLCV ingestion via `yfinance` with local CSV caching
- Synthetic data generator for offline testing
- Time-series aware train / validation / test splits (no data leakage)
- Market regime labeling (bull / bear / neutral)

### Feature Engineering (50+ signals)
| Category | Features |
|---|---|
| **Momentum** | RSI (14/28), MACD, Rate-of-Change (10/21d) |
| **Trend** | SMA/EMA ratios (5/10/20/50/200d), ADX |
| **Volatility** | Bollinger Bands, ATR, Realized Volatility (21/63d) |
| **Volume** | OBV, VWAP ratio, Volume ratio |
| **Price Structure** | High-low range, Close position, Gap |
| **Autoregressive** | Lagged returns & volatility (1/2/3/5d) |

### Models
| Model | Architecture | Notes |
|---|---|---|
| `random_forest` | 500-tree ensemble, balanced class weights | Robust baseline |
| `xgboost` | Gradient boosting, L1/L2 regularization | Strong performance |
| `lstm` | 2-layer LSTM + attention, cosine LR schedule | Sequence modeling |
| `ensemble` | Stacked meta-learner (logistic regression) | Best overall |

### Backtesting Engine
- Signal-threshold entry/exit logic (long-only or long-short)
- Transaction cost modeling (commissions + slippage in basis points)
- Full performance metrics suite:
  - Sharpe, Sortino, Calmar ratios
  - Max drawdown, Win rate, Profit factor
  - Information ratio vs. benchmark
- Trade log with per-trade PnL tracking
- Matplotlib visualization: equity curve, drawdown, return distribution, rolling Sharpe

---

## Quickstart

### Installation

```bash
git clone https://github.com/yourusername/quant-ml.git
cd quant-ml
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Default: SPY, 2015–2024, ensemble model
python main.py

# Custom ticker and model
python main.py --ticker AAPL --start 2018-01-01 --end 2024-01-01 --model xgboost --plot

# Long-short strategy with $500k
python main.py --ticker QQQ --model ensemble --initial-capital 500000 --plot
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--ticker` | `SPY` | Equity ticker symbol |
| `--start` | `2015-01-01` | Start date |
| `--end` | `2024-01-01` | End date |
| `--model` | `ensemble` | Model type (`xgboost`, `lstm`, `random_forest`, `ensemble`) |
| `--initial-capital` | `100000` | Starting portfolio value ($) |
| `--plot` | `False` | Generate and save result plots |

---

## Project Structure

```
quant-ml/
├── main.py          # CLI entry point
├── pipeline.py      # End-to-end orchestration
├── data.py          # Data ingestion & caching
├── features.py      # Technical indicator library
├── models.py        # XGBoost, LSTM, Random Forest, Ensemble
├── backtest.py      # Event-driven backtesting engine
├── requirements.txt
├── data/
│   └── cache/       # Cached OHLCV CSVs
├── results/         # Saved plots and output
└── notebooks/       # Exploratory analysis
```

---

## Sample Output

```
============================================================
  QuantML Pipeline  |  SPY  |  ENSEMBLE
  Period: 2015-01-01 → 2024-01-01
============================================================

[1/5] Fetching market data...
[2/5] Engineering features...
   Train: 1260 | Val: 270 | Test: 270 samples
[3/5] Training ensemble model...
  Fitting xgboost...
  Fitting random_forest...
[4/5] Evaluating model...
   Test Accuracy : 0.5741
   Test ROC-AUC  : 0.6123
[5/5] Running backtest...

==================================================
  BACKTEST RESULTS
==================================================
  Total Return (%)               38.72
  Annualized Return (%)          17.41
  Sharpe Ratio                   1.284
  Sortino Ratio                  1.891
  Max Drawdown (%)              -12.34
  Calmar Ratio                   1.411
  Win Rate (%)                   54.20
  Profit Factor                  1.342
  Total Trades                   87
  Avg Trade PnL ($)              445.06
  Information Ratio              0.623
  Benchmark Return (%)           22.15
==================================================

Top 10 Features:
  realized_vol_21                ██████████████ 0.0712
  bb_bandwidth                   ████████████  0.0601
  rsi_14                         ██████████    0.0487
  macd_hist                      █████████     0.0445
  ret_5d                         ████████      0.0398
  ...
```

---

## Methodology

### Target Variable
The model predicts **5-day forward return direction** (binary: up/down) rather than magnitude, which is more robust to noise in short-term price forecasting.

### Preventing Data Leakage
- Strict chronological splits — test data never seen during training or feature normalization
- Features are computed only from past data (no look-ahead bias)
- StandardScaler fitted only on training set, applied to val/test

### Walk-Forward Validation
For production deployment, replace the single split with a rolling walk-forward backtest by calling `run_pipeline()` with sliding windows to validate out-of-sample robustness across different market regimes.

---

## Extending the Project

- **Add tickers**: Run on any equity or ETF supported by Yahoo Finance
- **Add features**: Extend `features.py` with fundamental data, options flow, or sentiment
- **Add models**: Implement `BaseModel` interface to plug in Transformer, LightGBM, etc.
- **Portfolio optimization**: Combine signals across multiple assets with mean-variance optimization
- **Live trading**: Replace `data.py` with a broker API (Alpaca, Interactive Brokers) for paper trading

---

## Disclaimer

This project is for educational and research purposes only. Nothing in this repository constitutes financial advice. Past backtest performance does not guarantee future results. All trading involves risk of loss.

---

## License

MIT License — see `LICENSE` for details.
