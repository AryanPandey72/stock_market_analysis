import pandas as pd
from pandas import Series,DataFrame
import numpy as np

#Visualisation imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
#To grab stock data
from pandas_datareader import DataReader
from datetime import datetime

tech_list = ['AAPL','GOOGL','MSFT','AMZN']

end = datetime.now()

#Start date set to 1 year back
start = datetime(end.year-1,end.month,end.day)

import datetime
import yfinance as yf
# Date range
start = datetime.datetime(2020, 1, 1)
end = datetime.datetime(2024, 1, 1)

# Fetch data directly with yfinance
for stock in tech_list:
    try:
        data = yf.download(stock, start=start, end=end)
        globals()[stock] = data
        print(f"Data retrieved for {stock}")
    except Exception as e:
        print(f"Failed to retrieve data for {stock}: {e}")

AAPL.head()

AAPL.describe()
AAPL.info()
AAPL['Adj Close'].plot(legend=True,figsize=(12,5))
AAPL['Volume'].plot(legend=True,figsize=(12,5))

ma_day = [10,20,50]

for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))

    AAPL[column_name] = AAPL['Adj Close'].rolling(window=ma,center=False).mean()

AAPL.tail()
AAPL[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(12,5))

AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
AAPL['Daily Return'].tail()
AAPL['Daily Return'].plot(figsize=(14,5),legend=True,linestyle='--',marker='o')
sns.histplot(x=AAPL['Daily Return'].dropna(),bins=100,color='red')
import yfinance as yf
from pandas_datareader import data as pdr
import pandas as pd

# Use yfinance to download data directly
close_df = pd.DataFrame()  # Initialize an empty DataFrame

for stock in tech_list:
    try:
        data = yf.download(stock, start=start, end=end)
        adj_close = data['Adj Close']

        # Check if adj_close is a scalar and convert to a Series if needed
        if not isinstance(adj_close, (pd.Series, pd.DataFrame)):
            adj_close = pd.Series(adj_close, index=[data.index[0]]) # Create a Series with a single value

        close_df[stock] = adj_close
        print(f"Data retrieved for {stock}")
    except Exception as e:
        print(f"Failed to retrieve data for {stock}: {e}")
close_df.tail()
rets_df = close_df.pct_change()
rets_df.tail()
sns.jointplot(x='GOOGL', y='GOOGL', data=rets_df, kind='scatter', color='green')
sns.jointplot(x='GOOGL', y='AAPL', data=rets_df, kind='scatter')
sns.pairplot(rets_df.dropna())
sns.heatmap(rets_df.dropna(),annot=True)
rets = rets_df.dropna()
plt.figure(figsize=(8,5))

plt.scatter(rets.mean(),rets.std(),s=25)

plt.xlabel('Expected Return')
plt.ylabel('Risk')


#For adding annotatios in the scatterplot
for label,x,y in zip(rets.columns,rets.mean(),rets.std()):
    plt.annotate(
    label,
    xy=(x,y),xytext=(-120,20),
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    arrowprops = dict(arrowstyle='->',connectionstyle = 'arc3,rad=-0.5'))
sns.histplot(x=AAPL['Daily Return'].dropna(),bins=100,color='purple')
rets.head()
#Using Pandas built in qualtile method
rets['AAPL'].quantile(0.05)
days = 365

#delta t
dt = 1/365

mu = rets.mean()['GOOGL']

sigma = rets.std()['GOOGL']
#Function takes in stock price, number of days to run, mean and standard deviation values
def stock_monte_carlo(start_price,days,mu,sigma):

    price = np.zeros(days)
    price[0] = start_price

    shock = np.zeros(days)
    drift = np.zeros(days)

    for x in range(1,days):

        #Shock and drift formulas taken from the Monte Carlo formula
        shock[x] = np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))

        drift[x] = mu * dt

        #New price = Old price + Old price*(shock+drift)
        price[x] = price[x-1] + (price[x-1] * (drift[x]+shock[x]))

    return price
GOOGL.head()
start_price = 622.049 #Taken from above

for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))

plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Google')
runs = 10000

simulations = np.zeros(runs)

for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
q = np.percentile(simulations,1)

plt.hist(simulations,bins=200)

plt.figtext(0.6,0.8,s="Start price: $%.2f" %start_price)

plt.figtext(0.6,0.7,"Mean final price: $%.2f" % simulations.mean())

plt.figtext(0.6,0.6,"VaR(0.99): $%.2f" % (start_price -q,))

plt.figtext(0.15,0.6, "q(0.99): $%.2f" % q)

plt.axvline(x=q, linewidth=4, color='r')

plt.title(u"Final price distribution for Google Stock after %s days" %days, weight='bold')
