# Crypto Portfolio Strategies

## Overview

This project presents a Python-based framework for managing cryptocurrency portfolios in the context of Decentralized Finance (DeFi). Unlike traditional stock portfolios, crypto portfolios must contend with higher volatility, rapid market shifts, and a unique set of liquidity and momentum characteristics. This codebase aims to address these challenges through adaptive volatility-based allocation, momentum filtering, and multi-factor approaches tailored specifically for the crypto market.

Central to the philosophy of these strategies is the understanding that cryptocurrencies do not behave like traditional equities. Price dynamics can be more erratic, heavily influenced by market sentiment, protocol updates, regulatory news, and liquidity conditions unique to DeFi. While stock portfolios might rely on well-established risk models and historically stable correlations, crypto portfolios must incorporate dynamic tools that respond to high-frequency changes in volatility and momentum indicators. This code is designed with these distinctions in mind, providing strategies that adapt positions, maintain stablecoin buffers, and employ robust risk management metrics—such as drawdowns, Value at Risk (VaR), and Conditional VaR (CVaR)—more suited to the extreme conditions often observed in DeFi markets.

### Models Included:
1. **Model 1: Threshold-Based Volatility Management**
2. **Model 2: Inverse Volatility Weighted Portfolio**
3. **Model 3: Adaptive Multi-Factor Crypto Portfolio Strategy**
4. **Model Compare**: A script that allows side-by-side comparison of all three models.

### Why These Models are Unique to DeFi:
These models rely on stablecoins and technical indicators suited for crypto’s high volatility. Traditional stock portfolios often trust in more stable volatility regimes and do not commonly integrate stable, on-chain assets for risk-off periods. The dynamic switching and volatility thresholds are far more critical in a DeFi environment than in most equity market strategies.

---

## Getting Started

### Prerequisites

Ensure you have the following Python libraries installed:

```bash
pip install pandas numpy yfinance matplotlib ta
```

### Files

1. **`model1.py`**  
   Implements a volatility-based management strategy that adjusts cash allocation based on average market volatility and incorporates momentum filtering using short- and long-term Simple Moving Averages (SMAs).

2. **`model2.py`**  
   Implements an inverse volatility weighted portfolio strategy. It assigns weights to assets based on their historical volatility, allocating higher weights to less volatile assets.

3. **`model3.py`**  
   A multi-factor strategy that combines momentum and Relative Strength Index (RSI) indicators to select assets dynamically. It includes position sizing based on inverse volatility and enforces drawdown limits.

4. **`model_compare.py`**  
   Compares the results of all three models in terms of portfolio value, Sharpe Ratio, maximum drawdown, and other metrics. Users can tweak parameters such as start date, end date, and market conditions.

---

## Usage

### Run Individual Models
You can execute each model individually to understand its behavior and performance. For example:

```bash
python model1.py
```

Each script outputs:
- Final portfolio value
- Sharpe Ratio
- Maximum drawdown
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Plots for portfolio performance and drawdown

### Compare Models
To compare all models side by side, use `model_compare.py`:

```bash
python model_compare.py
```

This script:
- Downloads the necessary data for all models
- Executes each model
- Displays a comparison of portfolio value, drawdown, and key metrics
- Generates visualizations for easier comparison

---

## Customization

### Adjust Market Conditions
Modify the following parameters at the beginning of each script:
- `start_date`: Start date for historical data.
- `end_date`: End date for historical data.
- `rebalance_frequency`: Frequency of portfolio rebalancing (e.g., `'D'`, `'W'`, `'M'`).

### Adjust Portfolio Parameters
Each model has unique parameters:
- **Model 1**:
  - `low_volatility_threshold`
  - `high_volatility_threshold`
  - `min_cash_buffer`

- **Model 2**:
  - The weights are automatically adjusted based on inverse volatility.

- **Model 3**:
  - `max_position_size`: Maximum allocation to any single asset.
  - `max_total_drawdown`: Maximum portfolio drawdown limit.

---

## Performance Metrics

For all models, the following metrics are calculated:
- **Final Portfolio Value**: The total portfolio value at the end of the period.
- **Sharpe Ratio**: Measures risk-adjusted returns.
- **Maximum Drawdown**: The largest peak-to-trough loss over the evaluation period.
- **Value at Risk (VaR)**: The worst expected loss at a given confidence level (e.g., 95%).
- **Conditional Value at Risk (CVaR)**: The expected loss given that the portfolio has breached the VaR threshold.

---

## Example Outputs

### Individual Model Outputs
**Model 1:**
```
Final Portfolio Value: $50,774.58
Sharpe Ratio: 1.28
Maximum Drawdown: -28.32%
VaR (95%): -2.31%
CVaR (95%): -5.41%
```

**Model 2:**
```
Final Portfolio Value: $90,459.20
Sharpe Ratio: 0.96
Maximum Drawdown: -74.22%
Annualized Volatility: 63.63%
```

**Model 3:**
```
Final Portfolio Value: $194,330.93
Sharpe Ratio: 1.67
Maximum Drawdown: -29.45%
VaR (95%): -2.31%
CVaR (95%): -5.41%
```

---

## License

This project is open-source and available under the MIT License.