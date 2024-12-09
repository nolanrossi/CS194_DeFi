import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# backtest parameters
assets = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'USDC-USD']  #USDC acts as stable asset
start_date = '2020-01-01'
end_date = '2023-10-01'
initial_balance = 10000  #starting balance
low_volatility_threshold = 0.02
high_volatility_threshold = 0.05
min_cash_buffer = 0.1  #10% cash buffer

# Step 1: Download data
data = yf.download(assets, start=start_date, end=end_date)['Adj Close']

# Step 2: Calculate daily returns and rolling volatility
returns = data.pct_change().dropna()
volatility = returns.rolling(window=30).std() * np.sqrt(252)  #annualized volatility

#short (10-day) and long (50-day) simple moving average (SMAs) for momentum
short_sma = data.rolling(window=10).mean()
long_sma = data.rolling(window=50).mean()

# Step 3: threshold based volatility management and refined momentum filtering
def calculate_weights(volatility, data, short_sma, long_sma, low_volatility_threshold, high_volatility_threshold, min_cash_buffer):
    weights = pd.DataFrame(index=volatility.index, columns=volatility.columns)
    for date in volatility.index:

        if date not in short_sma.index or date not in long_sma.index:
            continue

        avg_volatility = volatility.loc[date].mean()

        # allocation of cash based on volatility thresholds
        if avg_volatility > high_volatility_threshold:
            cash_allocation = 0.5
        elif avg_volatility > low_volatility_threshold:
            cash_allocation = 0.2
        else:
            cash_allocation = min_cash_buffer  #set minimum cash buffer in low volatility

        # Momentum based filter: allocate only to assets with strong momentum (10-day > 50-day + margin)
        momentum_filter = (short_sma.loc[date] > long_sma.loc[date] * 1.05).astype(float)
        filtered_weights = momentum_filter / momentum_filter.sum()  # Normalize after filtering

        # Adjust for cash allocation
        weight = filtered_weights * (1 - cash_allocation)
        weight['Cash'] = cash_allocation

        # Apply weights
        weights.loc[date] = weight
    return weights

# Step 4: calculate weights at weekly intervals
rebalance_frequency = 'W'
weekly_volatility = volatility.resample(rebalance_frequency).last()
weekly_data = data.resample(rebalance_frequency).last()

# Add a cash column to the weekly data for tracking purposes
weekly_data['Cash'] = 1  # Cash has a fixed "price" of 1 USD

# Track portfolio value over time
portfolio_value_series = pd.Series(index=weekly_volatility.index)

# Calculate weights with threshold-based volatility management and momentum filtering
weights = calculate_weights(weekly_volatility, weekly_data, short_sma.resample(rebalance_frequency).last(), 
                            long_sma.resample(rebalance_frequency).last(), 
                            low_volatility_threshold, high_volatility_threshold, min_cash_buffer)

# Step 5: calculate portfolio returns by Applying weights to weekly returns 
weekly_returns = returns.resample(rebalance_frequency).apply(lambda x: (1 + x).prod() - 1)
weekly_returns['Cash'] = 0  # Cash has zero returns

weighted_returns = (weekly_returns * weights.shift()).sum(axis=1)

#portfolio value over time, explicitly casting to float to avoid warnings
portfolio_value = (1 + weighted_returns).cumprod() * float(initial_balance)
portfolio_value_series.loc[portfolio_value.index] = portfolio_value


#Max drawdown and sharpe
rolling_max = portfolio_value.cummax()
drawdown = (portfolio_value - rolling_max) / rolling_max
sharpe_ratio = weighted_returns.mean() / weighted_returns.std() * np.sqrt(52)  # Annualized for weekly returns

# Value at Risk (VaR) at 95% confidence level
VaR_95 = weighted_returns.quantile(0.05)

# Conditional Value at Risk (CVaR) at 95% confidence level
CVaR_95 = weighted_returns[weighted_returns <= VaR_95].mean()

# Print Metrics
print("Final Portfolio Value:", portfolio_value.iloc[-1])
print("Sharpe Ratio:", sharpe_ratio)
print("Maximum Drawdown:", drawdown.min())
print("Value at Risk (95%):", VaR_95)
print("Conditional Value at Risk (95%):", CVaR_95)

# Plot Portfolio Value over Time
plt.figure(figsize=(10, 6))
plt.plot(portfolio_value, label="Portfolio Value")
plt.title("Portfolio Value Over Time with Threshold-Based Volatility Management and Stable Asset")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.legend()
plt.show()

# Plot Drawdown over Time
plt.figure(figsize=(10, 6))
plt.plot(drawdown, color='red', label="Drawdown")
plt.title("Portfolio Drawdown Over Time")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend()
plt.show()
