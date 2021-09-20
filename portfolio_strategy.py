import numpy as np
import pandas as pd
import plotly
import math
from pandas_datareader import data as wb

def import_stock_prices(tickers, start, end=None) :
    """
    Import stock prices from yahoo
    """

    stock_prices = pd.DataFrame()

    for t in tickers :
        stock_prices[t] = wb.DataReader(t, data_source='yahoo', start=start, end=end)['Adj Close']
    
    return stock_prices

def simple_return(col) :
    """
    Compute simple daily return for a given column
    """
    return((col.pct_change()))

def cumulative_perc(col) :
    """
    Compute cumulative percentage for a given simple_perc column 
    """
    return ((1 + col).cumprod() - 1)

def moy_return(returns_df) :
    """
    Compute mean return on every dates for a given simple_perc column 
    """
    return returns_df.mean()*returns_df.count()

def portfolio_dataframe(tickers, start, end=None) : 
    """
    Compute the entire portfolio dataframe with cumulative percentages for each columns
    """

    new_data = pd.DataFrame()
    for t in tickers :
        new_data[t] = wb.DataReader(t, data_source='yahoo', start=start, end=end)['Adj Close']
        new_data[f'cumulative_perc_{t}'] = cumulative_perc(simple_return(new_data[t]))

    return new_data[[f'cumulative_perc_{t}' for t in tickers]]

def risk_calculation(daily_returns_df, weights) :
    """
    Compute the risk of a portfolio. (volatility)
    """
    #Weights transposed * (Covariance matrix * Weights)
    matrix_covariance = daily_returns_df.cov()*daily_returns_df.count()
    portfolio_variance = np.dot(weights.T,np.dot(matrix_covariance, weights))
    portfolio_std = np.sqrt(portfolio_variance)

    return portfolio_std

def sharpe_ratio(mean_return,  std, RF=0) :
    """
    Compute the sharpe ratio of a portfolio
    """
    sharpe_ratio = (mean_return-RF)/std
    return sharpe_ratio



def compute_portfolio_metrics(stock_prices, tickers, weights) :

    """
    Given stocks name & weights, this function compute the portfolio metrics (risk, return, sharpe..)
    """
    
    stock_daily_returns = pd.DataFrame()

    #Import stock prices of given stock names
    for t in tickers :
        stock_daily_returns[f'simple_return_{t}'] = simple_return(stock_prices[t])
        mean_returns = moy_return(stock_daily_returns)

    #Calculate metrics for one strategy

    if len(tickers)==1 :
        #Portfolio return calculation
        portfolio_return = mean_returns
        #Portfolio risks calculation
        portfolio_risk = np.sqrt(stock_daily_returns[f'simple_return_{tickers[0]}'].var()*stock_daily_returns.count())
        #Sharpe ratio
        sharpe_r = sharpe_ratio(portfolio_return, portfolio_risk, RF=0)
    else :
        #Portfolio return calculation
        portfolio_return = np.sum(weights*mean_returns)
        #Portfolio risks calculation
        portfolio_risk = risk_calculation(stock_daily_returns,weights)
        #Sharpe ratio
        sharpe_r = sharpe_ratio(portfolio_return, portfolio_risk, RF=0)

    return {'portfolio_returns':portfolio_return,'portfolio_weights':weights,'portfolio_risks':portfolio_risk,'sharpe_ratios':sharpe_r}


def random_strategies(stock_prices, tickers, N) :
    """
    Given stocks name, N = number of portfolios, this function compute random strategies.
    """
    final_metrics = {'portfolio_returns':[],'portfolio_weights':[],'portfolio_risks':[],'sharpe_ratios':[]}

    for portfolio in range(N) :
        #Generate random weights
        weights = np.random.random_sample(len(tickers))
        weights = np.round((weights/np.sum(weights)),len(tickers))
        dict_metrics = compute_portfolio_metrics(stock_prices=stock_prices, tickers=tickers, weights=weights)

        for key in dict_metrics.keys() :
            final_metrics[key].append(dict_metrics[key])

    return final_metrics


