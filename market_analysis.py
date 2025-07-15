import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime

# Seaborn styling
sns.set_style('whitegrid')

# Tech stock list
tech_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

# Date range
start = datetime(2020, 1, 1)
end = datetime(2024, 1, 1)

# Download data using yfinance with clean column structure
for stock in tech_list:
    try:
        data = yf.download(stock, start=start, end=end, group_by='column', auto_adjust=False)
        globals()[stock] = data
        print(f"Data retrieved for {stock}")
    except Exception as e:
        print(f"Failed to retrieve data for {stock}: {e}")

# -- AAPL Analysis --

# Preview AAPL
print(AAPL.head())
print(AAPL.describe())
print(AAPL.info())

# Plot Adjusted Close and Volume
AAPL['Adj Close'].plot(legend=True, figsize=(12, 5), title='AAPL Adjusted Close Price')
plt.show()

AAPL['Volume'].plot(legend=True, figsize=(12, 5), title='AAPL Trading Volume')
plt.show()

# Moving Averages
ma_day = [10, 20, 50]
for ma in ma_day:
    AAPL[f"MA for {ma} days"] = AAPL['Adj Close'].rolling(window=ma).mean()

AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(figsize=(12, 5), title='AAPL Moving Averages')
plt.show()

# Daily Returns
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
AAPL['Daily Return'].plot(figsize=(14, 5), legend=True, linestyle='--', marker='o', title='AAPL Daily Returns')
plt.show()

sns.histplot(AAPL['Daily Return'].dropna(), bins=100, color='red')
plt.title('AAPL Daily Return Distribution')
plt.show()

# Create adjusted close prices dataframe
close_df = pd.DataFrame()

for stock in tech_list:
    try:
        data = yf.download(stock, start=start, end=end, group_by='column', auto_adjust=False)
        close_df[stock] = data['Adj Close']
    except Exception as e:
        print(f"Failed to retrieve data for {stock}: {e}")

# Daily percentage returns
rets_df = close_df.pct_change()
print(rets_df.tail())

# Correlation plots
sns.jointplot(x='GOOGL', y='GOOGL', data=rets_df, kind='scatter', color='green')
sns.jointplot(x='GOOGL', y='AAPL', data=rets_df, kind='scatter')
sns.pairplot(rets_df.dropna())
plt.show()

# Risk vs Return
plt.figure(figsize=(8, 5))
plt.scatter(rets_df.mean(), rets_df.std(), s=25)
plt.xlabel('Expected Return')
plt.ylabel('Risk')
plt.title('Risk vs Return')

# Annotate points
for label, x, y in zip(rets_df.columns, rets_df.mean(), rets_df.std()):
    plt.annotate(label, xy=(x, y), xytext=(-120, 20),
                 textcoords='offset points', ha='right', va='bottom',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.5'))
plt.show()

# Value at Risk (VaR) for AAPL
print("AAPL VaR (0.05 quantile):", rets_df['AAPL'].quantile(0.05))

# -- Monte Carlo Simulation for GOOGL --

# Parameters
days = 365
dt = 1 / days
mu = rets_df.mean()['GOOGL']
sigma = rets_df.std()['GOOGL']
start_price = float(GOOGL['Adj Close'].iloc[-1])  # Last known price

# Monte Carlo simulation function
def stock_monte_carlo(start_price, days, mu, sigma):
    price = np.zeros(days)
    price[0] = start_price
    for x in range(1, days):
        shock = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        drift = mu * dt
        price[x] = price[x - 1] + (price[x - 1] * (drift + shock))
    return price

# Plot 100 simulations
for run in range(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Simulation - GOOGL')
plt.show()

# Histogram of 10,000 simulations
runs = 10000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price, days, mu, sigma)[-1]

q = np.percentile(simulations, 1)

plt.hist(simulations, bins=200)
plt.axvline(x=q, linewidth=4, color='r')

plt.figtext(0.6, 0.8, f"Start price: ${start_price:.2f}")
plt.figtext(0.6, 0.7, f"Mean final price: ${simulations.mean():.2f}")
plt.figtext(0.6, 0.6, f"VaR(0.99): ${start_price - q:.2f}")
plt.figtext(0.15, 0.6, f"q(0.99): ${q:.2f}")

plt.title(f"Final price distribution for GOOGL after {days} days", weight='bold')
plt.show()
