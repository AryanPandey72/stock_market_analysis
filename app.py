import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime

# Set seaborn style and page config
sns.set_style('whitegrid')
st.set_page_config(page_title="Tech Stocks Dashboard", layout="wide")

# Title
st.title("📈 Tech Stock Analysis & Simulation")
st.markdown("Analyze historical trends, risk metrics, and future predictions using Monte Carlo simulation.")

# Sidebar
tech_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
selected_stock = st.sidebar.selectbox("Select Stock", tech_list)
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2024, 1, 1))
ma_days = st.sidebar.multiselect("Select Moving Averages", [10, 20, 50], default=[10, 20, 50])

# Fetch data
@st.cache_data
def get_data(stock, start, end):
    return yf.download(stock, start=start, end=end, auto_adjust=False)

stock_data = get_data(selected_stock, start_date, end_date)

# Basic Info
st.subheader(f"📊 {selected_stock} Data Overview")
st.dataframe(stock_data.head())

# Descriptive Statistics
with st.expander("📈 Show Descriptive Statistics"):
    st.write(stock_data.describe())

# Adjusted Close Plot
st.subheader(f"📉 Adjusted Close Price - {selected_stock}")
fig, ax = plt.subplots(figsize=(12, 4))
stock_data['Adj Close'].plot(ax=ax, title=f"{selected_stock} Adjusted Close Price")
st.pyplot(fig)

# Volume Plot
st.subheader(f"📊 Trading Volume - {selected_stock}")
fig, ax = plt.subplots(figsize=(12, 4))
stock_data['Volume'].plot(ax=ax, title=f"{selected_stock} Volume")
st.pyplot(fig)

# Moving Averages
for ma in ma_days:
    stock_data[f"MA {ma}"] = stock_data['Adj Close'].rolling(window=ma).mean()

st.subheader("📏 Moving Averages")
fig, ax = plt.subplots(figsize=(12, 5))
stock_data[['Adj Close'] + [f"MA {ma}" for ma in ma_days]].plot(ax=ax, title=f"{selected_stock} Moving Averages")
st.pyplot(fig)

# Daily Return
stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()

st.subheader("📈 Daily Return")
fig, ax = plt.subplots(figsize=(12, 4))
stock_data['Daily Return'].plot(ax=ax, legend=True, linestyle='--', marker='o', title='Daily Returns')
st.pyplot(fig)

# Daily Return Distribution
st.subheader("📊 Daily Return Distribution")
fig, ax = plt.subplots()
sns.histplot(stock_data['Daily Return'].dropna(), bins=100, color='red', ax=ax)
ax.set_title('Daily Return Histogram')
st.pyplot(fig)

# Comparison Table for All Stocks
st.subheader("📋 Daily Returns Comparison")
@st.cache_data
def get_all_returns(stocks, start, end):
    close_df = pd.DataFrame()
    for stock in stocks:
        try:
            data = yf.download(stock, start=start, end=end, auto_adjust=False)
            if not data.empty:
                # Handle normal and multi-index column cases
                if 'Adj Close' in data.columns:
                    close_df[stock] = data['Adj Close']
                elif ('Adj Close', '') in data.columns:
                    close_df[stock] = data[('Adj Close', '')]
                else:
                    st.warning(f"'Adj Close' not found for {stock}")
            else:
                st.warning(f"No data for {stock}")
        except Exception as e:
            st.error(f"Error downloading {stock}: {e}")
    return close_df.pct_change()


rets_df = get_all_returns(tech_list, start_date, end_date)
st.dataframe(rets_df.tail())

# Correlation and Pairplot
st.subheader("🔗 Correlation & Risk-Return Plot")
with st.expander("Show Pairplots and Jointplots"):
    sns.pairplot(rets_df.dropna())
    st.pyplot(plt.gcf())

    sns.jointplot(x='GOOGL', y='AAPL', data=rets_df, kind='scatter')
    st.pyplot(plt.gcf())

# Risk vs Return Plot
fig, ax = plt.subplots(figsize=(8, 5))
plt.scatter(rets_df.mean(), rets_df.std(), s=25)
plt.xlabel('Expected Return')
plt.ylabel('Risk')
plt.title('Risk vs Return')
for label, x, y in zip(rets_df.columns, rets_df.mean(), rets_df.std()):
    plt.annotate(label, xy=(x, y), xytext=(-120, 20),
                 textcoords='offset points', ha='right', va='bottom',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.5'))
st.pyplot(fig)

# Value at Risk
st.subheader("📉 Value at Risk (VaR)")
var_aapl = rets_df['AAPL'].quantile(0.05)
st.write(f"**AAPL VaR (0.05 quantile): {var_aapl:.4f}**")

# Monte Carlo Simulation for GOOGL
st.subheader("🎲 Monte Carlo Simulation - GOOGL")

days = 365
dt = 1 / days
mu = rets_df.mean()['GOOGL']
sigma = rets_df.std()['GOOGL']
start_price = float(get_data('GOOGL', start_date, end_date)['Adj Close'].iloc[-1])

def stock_monte_carlo(start_price, days, mu, sigma):
    price = np.zeros(days)
    price[0] = start_price
    for x in range(1, days):
        shock = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        drift = mu * dt
        price[x] = price[x - 1] + (price[x - 1] * (drift + shock))
    return price

fig, ax = plt.subplots()
for run in range(100):
    ax.plot(stock_monte_carlo(start_price, days, mu, sigma), alpha=0.2)
ax.set_title("100 Simulated GOOGL Price Paths")
ax.set_xlabel("Days")
ax.set_ylabel("Price")
st.pyplot(fig)

# Histogram of simulations
runs = 10000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price, days, mu, sigma)[-1]

q = np.percentile(simulations, 1)
fig, ax = plt.subplots()
ax.hist(simulations, bins=200)
ax.axvline(x=q, linewidth=4, color='r')
plt.figtext(0.6, 0.8, f"Start price: ${start_price:.2f}")
plt.figtext(0.6, 0.7, f"Mean final price: ${simulations.mean():.2f}")
plt.figtext(0.6, 0.6, f"VaR(0.99): ${start_price - q:.2f}")
plt.figtext(0.15, 0.6, f"q(0.99): ${q:.2f}")
ax.set_title(f"Final GOOGL Price Distribution After {days} Days")
st.pyplot(fig)
