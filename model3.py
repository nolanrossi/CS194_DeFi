# Adaptive Multi-Factor Crypto Portfolio Strategy

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# Parameters
assets = [
    'BTC-USD', 'ETH-USD', 'ADA-USD', 'BNB-USD', 'XRP-USD',
    'SOL-USD', 'DOT-USD', 'DOGE-USD', 'USDC-USD'
]
start_date = '2020-01-01'
end_date = '2023-10-01'
initial_balance = 10000
rebalance_frequency = 'W'
max_position_size = 0.15  # maximum of portfolio per asset
max_total_drawdown = 0.15  # Maximum portfolio drawdown

# Step 1: get data
print("Downloading historical data...")
data = yf.download(assets, start=start_date, end=end_date)['Adj Close']
data = data.fillna(method='ffill').dropna()

#Daily returns
returns = data.pct_change().dropna()

# Step 2: compute Indicators
print("Calculating indicators...")
ema_short = pd.DataFrame(index=data.index, columns=assets)
ema_long = pd.DataFrame(index=data.index, columns=assets)
rsi = pd.DataFrame(index=data.index, columns=assets)

for asset in assets:
    ema_short[asset] = EMAIndicator(close=data[asset], window=12).ema_indicator()
    ema_long[asset] = EMAIndicator(close=data[asset], window=26).ema_indicator()
    rsi[asset] = RSIIndicator(close=data[asset], window=14).rsi()

# Step 3: multi factor Asset Selection
def select_assets(date):
    selected_assets = []
    for asset in assets:
        if asset == 'USDC-USD':
            continue  # Exclude stablecoin from momentum checks for accuracy
        if date not in ema_short.index or date not in ema_long.index or date not in rsi.index:
            continue
        # condition of Momentum
        if ema_short.loc[date, asset] > ema_long.loc[date, asset]:
            #condition of RSI
            if rsi.loc[date, asset] > 55:
                selected_assets.append(asset)
    return selected_assets

# Step 4: Dynamic Position Sizing
def calculate_weights(date, selected_assets):
    #allocate all to USDC in edgecase of no other attractive assets
    if not selected_assets:
        weights = pd.Series(0, index=assets)
        weights['USDC-USD'] = 1.0
        return weights

    #inverse volatility for selected assets
    vol = returns[selected_assets].rolling(window=14).std().loc[date]
    inv_vol = 1 / vol
    weights = inv_vol / inv_vol.sum()

    #cap position sizes
    weights = weights.apply(lambda x: min(x, max_position_size))
    total_alloc = weights.sum()
    weights['USDC-USD'] = 1 - total_alloc


    # ensure all assets are accounted
    full_weights = pd.Series(0, index=assets)
    full_weights.update(weights)
    return full_weights

print("Starting backtesting...")
portfolio_value = pd.Series(index=returns.index)
portfolio_value.iloc[0] = initial_balance
weights_history = pd.DataFrame(index=returns.index, columns=assets)
max_portfolio_value = initial_balance
drawdown = pd.Series(index=returns.index)


rebalance_dates = returns.resample(rebalance_frequency).last().index

#initial weights
weights = pd.Series(0, index=assets)
weights['USDC-USD'] = 1.0
weights_history.iloc[0] = weights

for i, date in enumerate(returns.index[1:], start=1):
    if date in rebalance_dates:
        selected_assets = select_assets(date)
        weights = calculate_weights(date, selected_assets)
        weights_history.loc[date] = weights
    else:
        weights = weights_history.loc[:date].ffill().iloc[-1]

    # daily portfolio return
    daily_return = (returns.loc[date] * weights).sum()
    portfolio_value.iloc[i] = portfolio_value.iloc[i - 1] * (1 + daily_return)

    # maximum portfolio value
    if portfolio_value.iloc[i] > max_portfolio_value:
        max_portfolio_value = portfolio_value.iloc[i]

    #drawdown
    drawdown.iloc[i] = (portfolio_value.iloc[i] - max_portfolio_value) / max_portfolio_value

    # get maximum drawdown limit
    if drawdown.iloc[i] < -max_total_drawdown:
        print(f"Maximum drawdown limit reached on {date.date()}. Exiting positions.")
        # Liquidate positions to USDC
        weights = pd.Series(0, index=assets)
        weights['USDC-USD'] = 1.0
        weights_history.loc[date] = weights
        max_portfolio_value = portfolio_value.iloc[i]

#drop NaN's
portfolio_value = portfolio_value.dropna()
drawdown = drawdown.dropna()
weights_history = weights_history.dropna()

def calculate_performance_metrics(portfolio_value):
    returns_series = portfolio_value.pct_change().dropna()
    sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252)
    max_drawdown = (portfolio_value / portfolio_value.cummax() - 1).min()
    VaR_95 = returns_series.quantile(0.05)
    CVaR_95 = returns_series[returns_series <= VaR_95].mean()
    return sharpe_ratio, max_drawdown, VaR_95, CVaR_95

sharpe_ratio, max_drawdown_value, VaR_95, CVaR_95 = calculate_performance_metrics(portfolio_value)

# results
print("\nPerformance Metrics:")
print(f"Final Portfolio Value: ${portfolio_value.iloc[-1]:.2f}")
print(f"Total Return: {((portfolio_value.iloc[-1] / initial_balance - 1) * 100):.2f}%")
print(f"Annualized Return: {((portfolio_value.iloc[-1] / initial_balance) ** (252 / len(portfolio_value)) - 1) * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown_value:.2%}")
print(f"Value at Risk (95%): {VaR_95:.2%}")
print(f"Conditional Value at Risk (95%): {CVaR_95:.2%}")

# Step 5: visualization
print("\nGenerating plots...")
# Plot value
plt.figure(figsize=(12, 6))
plt.plot(portfolio_value, label='Adaptive Multi-Factor Strategy')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Plot Drawdown
plt.figure(figsize=(12, 6))
plt.plot(drawdown, label='Drawdown', color='red')
plt.title('Portfolio Drawdown Over Time')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.legend()
plt.grid(True)
plt.show()

# Plot Asset Allocation Over Time
weights_history.plot(kind='area', stacked=True, figsize=(12, 6))
plt.title('Portfolio Asset Allocation Over Time')
plt.xlabel('Date')
plt.ylabel('Allocation')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
