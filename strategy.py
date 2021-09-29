import numpy as np
import pandas as pd
import plotly
import math
from pandas_datareader import data as wb


class PortfolioStrategy:

    def __init__(self,
                 df_stocks : pd.DataFrame(),
                 weights : list()) :

        self.stock_prices = df_stocks
        self.weights = weights
        self.tickers = df_stocks.columns.to_list()
        self.stock_daily_returns = pd.DataFrame()
        self.stock_daily_returns = self.simple_return()
        self.stock_daily_returns.col = [f'simple_return_{t}' for t in self.tickers]

        self.mean_returns = self.moy_return()
    

    def simple_return(self) :
        """
        Compute simple daily return for a given column
        """
        return self.stock_prices.pct_change()

    def moy_return(self) :
        """
        Compute mean return on every dates for a given simple_perc column 
        """
        return self.stock_daily_returns.mean()*self.stock_daily_returns.count()

    def risk_calculation(self) :
        """
        Compute the risk of a portfolio. (volatility)
        """
        #Weights transposed * (Covariance matrix * Weights)
        matrix_covariance = self.stock_daily_returns.cov()*self.stock_daily_returns.count()
        portfolio_variance = np.dot(self.weights.T,np.dot(matrix_covariance, self.weights))
        portfolio_std = np.sqrt(portfolio_variance)

        return portfolio_std

    def sharpe_ratio(self, mean_return,  std, RF=0) :
        """
        Compute the sharpe ratio of a portfolio
        """
        sharpe_ratio = (mean_return-RF)/std

        return sharpe_ratio
    
    def compute_metrics(self) :
        """
        Given stocks name & weights, this function compute the portfolio metrics (risk, return, sharpe..)
        """
        #Calculate metrics for one strategy
        if len(self.tickers)==1 :
            #Portfolio return calculation
            portfolio_return = self.mean_returns
            #Portfolio risks calculation
            portfolio_risk = np.sqrt(self.stock_daily_returns[f'simple_return_{self.tickers[0]}'].var()*self.stock_daily_returns.count())
            #Sharpe ratio
            sharpe_r = self.sharpe_ratio(portfolio_return, portfolio_risk, RF=0)
        else :
            #Portfolio return calculation
            portfolio_return = np.sum(self.weights*self.mean_returns)
            #Portfolio risks calculation
            portfolio_risk = self.risk_calculation()
            #Sharpe ratio
            sharpe_r = self.sharpe_ratio(portfolio_return, portfolio_risk, RF=0)

        return {'portfolio_returns':portfolio_return,'portfolio_weights':self.weights,'portfolio_risks':portfolio_risk,'sharpe_ratios':sharpe_r}
