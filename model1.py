import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

#init currencies to analyze
cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']

#  timeframe
start_date = '2020-01-01'
end_date = '2023-10-01'


data = yf.download(cryptos, start=start_date, end=end_date)['Adj Close']

# daily returns
returns = data.pct_change().dropna()

# 30-day rolling volatility (this is annualized)
volatility = returns.rolling(window=30).std() * np.sqrt(252)

# invers of volatility
inv_volatility = 1 / volatility

#normalize weights so that they sum to 1
weights = inv_volatility.div(inv_volatility.sum(axis=1), axis=0)

#drop NaN values after initial rolling window
weights = weights.dropna()

#backtesting
def backtest_portfolio(returns, weights):
    weights_aligned = weights.shift(1).reindex(returns.index).fillna(method='ffill')
    portfolio_returns = (returns * weights_aligned).sum(axis=1)
    return portfolio_returns

vol_based_returns = backtest_portfolio(returns, weights) #volatility-based portfolio returns


equal_weights = pd.DataFrame(1/len(cryptos), index=returns.index, columns=cryptos) #Equally-weighted portfolio weights


equal_weighted_returns = backtest_portfolio(returns, equal_weights)#Equally-weighted portfolio returns


#Cumulative returns for both portfolios
vol_based_cum_returns = (1 + vol_based_returns).cumprod()
equal_weighted_cum_returns = (1 + equal_weighted_returns).cumprod()

# Plot  performance
plt.figure(figsize=(12, 6))
plt.plot(vol_based_cum_returns, label='Volatility-Based Portfolio')
plt.plot(equal_weighted_cum_returns, label='Equally-Weighted Portfolio')
plt.title('Portfolio Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

#performance metrics functions
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    # Calculate annualized Sharpe Ratio
    excess_return = returns.mean() * 252 - risk_free_rate
    std_dev = returns.std() * np.sqrt(252)
    sharpe_ratio = excess_return / std_dev
    return sharpe_ratio

def calculate_max_drawdown(cumulative_returns):
    cumulative_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_annualized_volatility(returns):
    return returns.std() * np.sqrt(252)

#Volatility-Based Portfolio Metrics
vb_sharpe = calculate_sharpe_ratio(vol_based_returns)
vb_max_dd = calculate_max_drawdown(vol_based_cum_returns)
vb_volatility = calculate_annualized_volatility(vol_based_returns)

#Equally-Weighted Portfolio Metrics
ew_sharpe = calculate_sharpe_ratio(equal_weighted_returns)
ew_max_dd = calculate_max_drawdown(equal_weighted_cum_returns)
ew_volatility = calculate_annualized_volatility(equal_weighted_returns)

# results
print("Volatility-Based Portfolio Metrics:")
print(f"Sharpe Ratio: {vb_sharpe:.2f}")
print(f"Maximum Drawdown: {vb_max_dd:.2%}")
print(f"Annualized Volatility: {vb_volatility:.2%}\n")

print("Equally-Weighted Portfolio Metrics:")
print(f"Sharpe Ratio: {ew_sharpe:.2f}")
print(f"Maximum Drawdown: {ew_max_dd:.2%}")
print(f"Annualized Volatility: {ew_volatility:.2%}")
