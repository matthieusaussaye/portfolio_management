from strategy import *
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

def portfolio_dataframe(tickers, start, end=None) : 
    """
    Compute the entire portfolio dataframe with cumulative percentages for each columns
    """

    new_data = pd.DataFrame()
    for t in tickers :
        new_data[t] = wb.DataReader(t, data_source='yahoo', start=start, end=end)['Adj Close']
        new_data[f'cumulative_perc_{t}'] = cumulative_perc(simple_return(new_data[t]))

    return new_data[[f'cumulative_perc_{t}' for t in tickers]]


def random_strategies(stock_prices, tickers, N) :
    """
    Given stocks name, N = number of portfolios, this function compute random strategies.
    """
    final_metrics = {'portfolio_returns':[],'portfolio_weights':[],'portfolio_risks':[],'sharpe_ratios':[]}

    for portfolio in range(N) :
        #Generate random weights

        weights = np.random.random_sample(len(tickers))
        weights = np.round((weights/np.sum(weights)),len(tickers))

        Pstrategy = PortfolioStrategy(df_stocks=stock_prices, weights=weights)
        dict_metrics = Pstrategy.compute_metrics()

        for key in dict_metrics.keys() :
            final_metrics[key].append(dict_metrics[key])

    return final_metrics