# Momentum-Mean-Reversion-Dual-Strategy-Backtester
This repository contains a Python implementation of a quantitative backtesting engine. It evaluates three distinct trading logics—Momentum, Mean-Reversion, and a Dual-Signal Agreement strategy—across a 10-year historical period for a multi-asset portfolio.

## Strategy Logic
The system processes three specific trading approaches:

Momentum (SMA Crossover): Utilizes the 50-day and 200-day Simple Moving Averages. It enters a Long position when the 50-day SMA is above the 200-day SMA and a Short position otherwise.

Mean-Reversion (RSI): Uses a 14-day Relative Strength Index. It enters a Long position when the RSI is below 30 (oversold) and a Short position when the RSI is above 70 (overbought).

Dual Strategy (Agreement): A filtering logic that only takes a position when both the Momentum and Mean-Reversion signals match. If the signals diverge, the strategy remains Flat (0).

## Technical Implementation
Data Handling: Automatic downloading of adjusted daily prices via Yahoo Finance for tickers including SPY, QQQ, GLD, USO, TLT, AAPL, and TSLA.

Bias Mitigation: Implements a 1-day position shift to ensure trades are executed based on prior-day closing data, preventing lookahead bias.

Transaction Costs: Incorporates a fixed cost of 0.05% per trade to provide a realistic net return profile.

Performance Metrics: Calculates four key KPIs:

Annualized Return (CAGR)

Annualized Volatility

Sharpe Ratio (Risk-adjusted return)

Maximum Drawdown (Peak-to-trough decline)

## Installation and Usage
Prerequisites
The following Python libraries are required:

Plaintext
yfinance
pandas
numpy
Execution
Run the script to generate a ticker-by-ticker performance breakdown in the console:

Bash
python backtest_script.py
