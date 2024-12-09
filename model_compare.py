# compare_crypto_models.py

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# ================================
# User-Defined Parameters
# ================================

#cryptocurrencies to analyze/include in each model
cryptos_model1 = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'USDC-USD']
cryptos_model2 = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
cryptos_model3 = [
    'BTC-USD', 'ETH-USD', 'ADA-USD', 'BNB-USD', 'XRP-USD',
    'SOL-USD', 'DOT-USD', 'DOGE-USD', 'USDC-USD'
]

#time period to model
start_date = '2020-01-01'
end_date = '2023-10-01'

# Market condition parameters for Model 1
low_volatility_threshold = 0.02
high_volatility_threshold = 0.05
min_cash_buffer = 0.1  # Maintain at least 10% cash buffer

#rebalancing
rebalance_frequency = 'W'  # 'D' (Daily), 'W' (Weekly), 'M' (Monthly)

#init
initial_balance = 10000  # Starting balance
max_position_size = 0.15  # maximum of portfolio per asset
max_total_drawdown = 0.15  # Maximum portfolio drawdown

# ===================================
# Functions
# ==================================

def get_data(assets, start_date, end_date):
    """Download historical data for given assets."""
    data = yf.download(assets, start=start_date, end=end_date)['Adj Close']
    data = data.ffill().dropna()
    return data

def model1():
    """Threshold-Based Volatility Management with Momentum Filtering."""
    print("\nRunning Model 1...")
    assets = cryptos_model1.copy()
    data = get_data(assets, start_date, end_date)
    returns = data.pct_change().dropna()
    volatility = returns.rolling(window=30).std() * np.sqrt(252)
    short_sma = data.rolling(window=10).mean()
    long_sma = data.rolling(window=50).mean()

    def calculate_weights(volatility, data, short_sma, long_sma):
        weights = pd.DataFrame(index=volatility.index, columns=volatility.columns)
        for date in volatility.index:
            if date not in short_sma.index or date not in long_sma.index:
                continue
            avg_volatility = volatility.loc[date].mean()
            if avg_volatility > high_volatility_threshold:
                cash_allocation = 0.5
            elif avg_volatility > low_volatility_threshold:
                cash_allocation = 0.2
            else:
                cash_allocation = min_cash_buffer
            momentum_filter = (short_sma.loc[date] > long_sma.loc[date] * 1.05).astype(float)
            if momentum_filter.sum() == 0:
                weight = pd.Series(0, index=volatility.columns)
                weight['USDC-USD'] = 1.0
            else:
                filtered_weights = momentum_filter / momentum_filter.sum()
                weight = filtered_weights * (1 - cash_allocation)
                weight['USDC-USD'] = cash_allocation
            weights.loc[date] = weight
        return weights

    # weights
    weights = calculate_weights(volatility, data, short_sma, long_sma)
    weights = weights.ffill().fillna(0).infer_objects()

    # portfolio returns
    portfolio_returns = (returns * weights.shift()).sum(axis=1)
    portfolio_value = (1 + portfolio_returns).cumprod() * initial_balance

    # performance Metrics
    rolling_max = portfolio_value.cummax()
    drawdown = (portfolio_value - rolling_max) / rolling_max
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    VaR_95 = portfolio_returns.quantile(0.05)
    CVaR_95 = portfolio_returns[portfolio_returns <= VaR_95].mean()

    #results
    results = {
        'Portfolio Value': portfolio_value,
        'Drawdown': drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': drawdown.min(),
        'VaR_95': VaR_95,
        'CVaR_95': CVaR_95,
    }

    print(f"Model 1 Final Portfolio Value: ${portfolio_value.iloc[-1]:.2f}")
    print(f"Model 1 Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Model 1 Maximum Drawdown: {drawdown.min():.2%}")
    return results

def model2():
    """Inverse Volatility Weighted Portfolio."""
    print("\nRunning Model 2...")
    assets = cryptos_model2.copy()
    data = get_data(assets, start_date, end_date)
    returns = data.pct_change().dropna()
    volatility = returns.rolling(window=30).std() * np.sqrt(252)
    inv_volatility = 1 / volatility
    weights = inv_volatility.div(inv_volatility.sum(axis=1), axis=0)
    weights = weights.ffill().fillna(0).infer_objects()

    portfolio_returns = (returns * weights.shift()).sum(axis=1)
    portfolio_value = (1 + portfolio_returns).cumprod() * initial_balance

    #performance Metrics
    cumulative_returns = (1 + portfolio_returns).cumprod()
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)

    # Store results
    results = {
        'Portfolio Value': portfolio_value,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Annualized Volatility': annualized_volatility,
    }

    print(f"Model 2 Final Portfolio Value: ${portfolio_value.iloc[-1]:.2f}")
    print(f"Model 2 Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Model 2 Maximum Drawdown: {max_drawdown:.2%}")
    return results

def model3():
    """Adaptive Multi-Factor Crypto Portfolio Strategy."""
    print("\nRunning Model 3...")
    assets = cryptos_model3.copy()
    data = get_data(assets, start_date, end_date)
    returns = data.pct_change().dropna()

    #indicators
    ema_short = data.apply(lambda x: EMAIndicator(close=x, window=12).ema_indicator())
    ema_long = data.apply(lambda x: EMAIndicator(close=x, window=26).ema_indicator())
    rsi = data.apply(lambda x: RSIIndicator(close=x, window=14).rsi())

    portfolio_value = pd.Series(index=returns.index)
    portfolio_value.iloc[0] = initial_balance
    weights_history = pd.DataFrame(index=returns.index, columns=assets)
    max_portfolio_value = initial_balance
    drawdown = pd.Series(index=returns.index)

    #rebalance intervals
    rebalance_dates = returns.resample(rebalance_frequency).last().index

    #initial weights
    weights = pd.Series(0, index=assets)
    weights['USDC-USD'] = 1.0
    weights_history.iloc[0] = weights

    for i, date in enumerate(returns.index[1:], start=1):
        if date in rebalance_dates:
            selected_assets = []
            #choose assets
            for asset in assets:
                if asset == 'USDC-USD':
                    continue
                if date not in ema_short.index or date not in ema_long.index or date not in rsi.index:
                    continue
                if pd.isna(ema_short.loc[date, asset]) or pd.isna(ema_long.loc[date, asset]) or pd.isna(rsi.loc[date, asset]):
                    continue
                if ema_short.loc[date, asset] > ema_long.loc[date, asset] and rsi.loc[date, asset] > 55:
                    selected_assets.append(asset)
            # weights
            if selected_assets:
                vol = returns[selected_assets].rolling(window=14).std().loc[date]
                inv_vol = 1 / vol
                weights_selected = inv_vol / inv_vol.sum()
                weights_selected = weights_selected.apply(lambda x: min(x, max_position_size))
                total_alloc = weights_selected.sum()
                weights = pd.Series(0, index=assets)
                weights.update(weights_selected)
                weights['USDC-USD'] = 1 - total_alloc
            else:
                weights = pd.Series(0, index=assets)
                weights['USDC-USD'] = 1.0
            weights_history.loc[date] = weights
        else:
            weights = weights_history.loc[:date].ffill().iloc[-1]

        # daily  return
        daily_return = (returns.loc[date] * weights).sum()
        portfolio_value.iloc[i] = portfolio_value.iloc[i - 1] * (1 + daily_return)

        # update top portfolio value
        if portfolio_value.iloc[i] > max_portfolio_value:
            max_portfolio_value = portfolio_value.iloc[i]

        # drawdown
        drawdown.iloc[i] = (portfolio_value.iloc[i] - max_portfolio_value) / max_portfolio_value

        if drawdown.iloc[i] < -max_total_drawdown:
            print(f"Model 3: Max drawdown limit reached on {date.date()}. Exiting positions.")
            weights = pd.Series(0, index=assets)
            weights['USDC-USD'] = 1.0
            weights_history.loc[date] = weights
            max_portfolio_value = portfolio_value.iloc[i]

    # Drop NaN's
    portfolio_value = portfolio_value.dropna()
    drawdown = drawdown.dropna()
    weights_history = weights_history.dropna()

    #performance 
    returns_series = portfolio_value.pct_change().dropna()
    sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252)
    max_drawdown_value = (portfolio_value / portfolio_value.cummax() - 1).min()
    VaR_95 = returns_series.quantile(0.05)
    CVaR_95 = returns_series[returns_series <= VaR_95].mean()

    #  results
    results = {
        'Portfolio Value': portfolio_value,
        'Drawdown': drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown_value,
        'VaR_95': VaR_95,
        'CVaR_95': CVaR_95,
    }

    print(f"Model 3 Final Portfolio Value: ${portfolio_value.iloc[-1]:.2f}")
    print(f"Model 3 Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Model 3 Maximum Drawdown: {max_drawdown_value:.2%}")
    return results

# ==================================
#   Execution
# ====================================

if __name__ == "__main__":
    # Run models
    results_model1 = model1()
    results_model2 = model2()
    results_model3 = model3()

    print("\nComparing Models...")

    # Plot Values
    plt.figure(figsize=(12, 6))
    plt.plot(results_model1['Portfolio Value'], label='Model 1')
    plt.plot(results_model2['Portfolio Value'], label='Model 2')
    plt.plot(results_model3['Portfolio Value'], label='Model 3')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Drawdowns
    plt.figure(figsize=(12, 6))
    plt.plot(results_model1['Drawdown'], label='Model 1')
    # Model 2 drawdiown
    cumulative_returns_model2 = results_model2['Portfolio Value']
    rolling_max_model2 = cumulative_returns_model2.cummax()
    drawdown_model2 = (cumulative_returns_model2 - rolling_max_model2) / rolling_max_model2
    plt.plot(drawdown_model2, label='Model 2')
    plt.plot(results_model3['Drawdown'], label='Model 3')
    plt.title('Drawdown Over Time')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Comparison of Performance Metrics
    print("\nPerformance Comparison:")
    metrics = pd.DataFrame({
        'Model 1': {
            'Final Portfolio Value': results_model1['Portfolio Value'].iloc[-1],
            'Sharpe Ratio': results_model1['Sharpe Ratio'],
            'Max Drawdown': results_model1['Max Drawdown'],
            'VaR_95': results_model1['VaR_95'],
            'CVaR_95': results_model1['CVaR_95'],
        },
        'Model 2': {
            'Final Portfolio Value': results_model2['Portfolio Value'].iloc[-1],
            'Sharpe Ratio': results_model2['Sharpe Ratio'],
            'Max Drawdown': results_model2['Max Drawdown'],
            'Annualized Volatility': results_model2['Annualized Volatility'],
        },
        'Model 3': {
            'Final Portfolio Value': results_model3['Portfolio Value'].iloc[-1],
            'Sharpe Ratio': results_model3['Sharpe Ratio'],
            'Max Drawdown': results_model3['Max Drawdown'],
            'VaR_95': results_model3['VaR_95'],
            'CVaR_95': results_model3['CVaR_95'],
        }
    })
    print(metrics)

    print("\nTo tweak market conditions, adjust the 'start_date', 'end_date', and other parameters at the beginning of the script and re-run it.")
