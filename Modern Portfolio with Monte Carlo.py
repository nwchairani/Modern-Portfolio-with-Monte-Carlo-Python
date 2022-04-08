"""
Modern Portfolio Theory with Monte Carlo
@author: Novia Widya Chairani

"""

import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as dt
import scipy.optimize as optimization

#################################################################################################

# Portfolio's stock list
# MMM=3M, ABT=Abbott, ACN=Accenture, ARE=Alexandria Real Estate, AMZN=Amazon
stocks = ['MMM','ABT','ACN','ARE','AMZN']

# Downloading the data from Yahoo! Finance
def download_data(stocks):
	data = web.DataReader(stocks, data_source='yahoo',start='31/01/2017',end='31/01/2022')['Adj Close'] #only get the closing price
	data.columns = stocks #set column names equal to stocks list
	return data

# Calling the data as dataframe
data= web.DataReader(stocks, data_source='yahoo',start='31/01/2017',end='31/01/2022')['Adj Close']

# Price series of each asset
# graph the price series
def show_data(data):
    data.plot(figsize=(10,5), title='Stock Price Series')
    plt.ylabel("Adjusted Closing Price")
    plt.show()
# use plot function
show_data(data)

# Log returns series of each asset
# calculate log daily returns
log_returns = np.log(data) - np.log(data.shift(1))

#################################################################################################

# Create a function to plot the price data above and call that function
# Use natural logarithm for normalization purposes
returns = np.log(data/data.shift(1))
	
# Print out mean and covariance of stocks within [start_date, end_date]. There are 252 trading days within a year
print(returns.mean()*252)#returns[] for individual stock returns
print(returns.cov()*252)#returns[] for individual stock returns

# Weights defines what stocks to include (with what portion) in the portfolio
weights = np.random.random(len(stocks))
weights /= np.sum(weights)
len(stocks)	

# Expected portfolio return
def port_return(weights):
    return np.sum(returns.mean()*weights)*252
    

# Expected portfolio variance
portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
print("Expected variance:", portfolio_variance)


def generate_portfolios(weights, returns):
    np.random.seed(2323)
    preturns = []
    pvariances = []

	#Monte-Carlo simulation: we generate several random weights -> so random portfolios 
    for i in range(100000):
        weights = np.random.random(len(stocks))
        weights/=np.sum(weights)# equivalent to weights = weights/sum(weights)
        preturns.append(np.sum(returns.mean()*weights)*252)
        pvariances.append(np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights))))
	
    preturns = np.array(preturns)
    pvariances = np.array(pvariances)
    return preturns,pvariances

# Call the generate_portfolios(weights, returns) function
preturns,pvariances=generate_portfolios(weights, returns)
preturns.mean()
pvariances.min()
def plot_portfolios(preturns, pvariances):
	plt.figure(figsize=(10,6))
	plt.scatter(pvariances,preturns,c=preturns/pvariances,marker='o')
	plt.grid(True)
	plt.xlabel('Expected Volatility')
	plt.ylabel('Expected Return')
	plt.colorbar(label='Sharpe Ratio')
	plt.show()
plot_portfolios(preturns,pvariances)

# OK this is the result of the simulation ... now we have to find the optimal portfolio with 
# some optimization technique - scipy can optimize functions (minimum/maximum finding)
def statistics(weights, returns):
	portfolio_return=np.sum(returns.mean()*weights)*252
	portfolio_volatility=np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights)))
	return np.array([portfolio_return,portfolio_volatility,portfolio_return/portfolio_volatility])

statistics(weights, returns)
# [1] Here, we want to maximize according to the volatility(risk)
# note: **2 squares the standard deviation to get the variance again
def	min_func_variance(weights, returns):
	return	statistics(weights, returns)[1] **2

# Get target return
target_return=np.median(preturns)

# Define target return constraint function
def portfolio_return(weights):
    return statistics(weights, returns)[0]

# Constraints (a) and (b)
constraintsa = [{'type':'eq','fun': lambda x: np.sum(x)-1}]#the sum of weights is 1
constraints = [{'type':'eq','fun': lambda x: np.sum(x)-1}, {'type':'eq','fun': lambda x: portfolio_return(x)-target_return}]#the sum of weights is 1  portfolio_return(x) x is the weight, constraint b is w=1 and expected portfolio target return equals [50] the median portfolio return
bounds = tuple((0,1) for x in range(len(stocks))) #the weights can be 1 at most: 1 when 100% of money is invested into a single stock

# Min var optimisation (a)
optimuma=optimization.minimize(fun=min_func_variance,x0=weights,args=returns,method='SLSQP',bounds=bounds,constraints=constraintsa)  # minimum negative SR

# Min var optimisation (b)
optimum=optimization.minimize(fun=min_func_variance,x0=weights,args=returns,method='SLSQP',bounds=bounds,constraints=constraints) # minimum negative SR with different constraint

print("Optimal weights (a):", optimuma['x'].round(3))
print("Expected return, volatility and Sharpe ratio:", statistics(optimuma['x'].round(3),returns))
print("Optimal weights (b):", optimum['x'].round(3))
print("Expected return, volatility and Sharpe ratio:", statistics(optimum['x'].round(3),returns))


# [2] Here, we want to maximize according to the Sharpe-ratio
# note: maximizing f(x) function is the same as minimizing -f(x) 
def	min_func_sharpe(weights,returns):
	return	-statistics(weights,returns)[2]

# Min -SR optimisation (a)
optimum_sra=optimization.minimize(fun=min_func_sharpe,x0=weights,args=returns,method='SLSQP',bounds=bounds,constraints=constraintsa) 
# Min -SR optimisation (b)
optimum_sr=optimization.minimize(fun=min_func_sharpe,x0=weights,args=returns,method='SLSQP',bounds=bounds,constraints=constraints) 

print("Optimal weights SR (a):", optimum_sra['x'].round(3))
print("Expected return, volatility and Sharpe ratio:", statistics(optimum_sra['x'].round(3),returns))
print("Optimal weights SR (b):", optimum_sr['x'].round(3))
print("Expected return, volatility and Sharpe ratio:", statistics(optimum_sr['x'].round(3),returns))

# Plotting
def show_optimal_portfolio(optimum, returns, preturns, pvariances):
    plt.figure(figsize=(10,6))
    plt.scatter(pvariances,preturns,c=preturns/pvariances,marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(optimum_sra['x'],returns)[1],statistics(optimum_sra['x'],returns)[0],
             'b*',markersize=14.0)
    plt.plot(statistics(optimum_sr['x'],returns)[1],statistics(optimum_sr['x'],returns)[0],
             'rs',markersize=10.0)
    plt.plot(statistics(optimum['x'],returns)[1],statistics(optimum['x'],returns)[0],
             'r*',markersize=14.0)
    plt.plot(statistics(optimuma['x'],returns)[1],statistics(optimuma['x'],returns)[0],
             'bs',markersize=10.0)
    plt.show()
show_optimal_portfolio(optimum, returns, preturns, pvariances)    













