   
import numpy as np
# setting the seed allows for reproducible results
np.random.seed(123)

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K

import pandas as pd

class Model:
    def __init__(self):
        self.data = None
        self.model = None
        
    def __build_model(self, input_shape, outputs):
        '''
        Builds and returns the Deep Neural Network that will compute the allocation ratios
        that optimize the Sharpe Ratio of the portfolio
        
        inputs: input_shape - tuple of the input shape, outputs - the number of assets
        returns: a Deep Neural Network model
        '''
        model = Sequential([ # create the model with input shape, output shape (10)
            LSTM(64, input_shape=input_shape),
            Flatten(),
            Dense(outputs, activation='softmax')
        ])

        def sharpe_loss(_, y_pred):

            # make all time-series start at 1
            data = tf.divide(self.data, self.data[0])  
            
            # value of the portfolio after allocations applied
            portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1) 
            portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % daily change formula
            sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns) #Daily sharpe - not annualized
            
            # since we want to maximize Sharpe, while gradient descent minimizes the loss, 
            # we can negate Sharpe (the min of a negated function is its max)
            return -sharpe
        
        model.compile(loss=sharpe_loss, optimizer='adam')

        return model
    
    def get_allocations(self, data: pd.DataFrame):
        '''
        Computes and returns the allocation ratios that optimize the Sharpe over the given data
        
        input: data - DataFrame of historical closing prices of various assets
        
        return: the allocations ratios for each of the given assets
        '''
        
        # data with returns : concatenate all in one vector : stock price then perc change
        data_w_ret = np.concatenate([data.values[1:], data.pct_change().values[1:]], axis=1)
        data = data.iloc[1:]

        self.data = tf.cast(tf.constant(data), float) #create a float tensor readable by tensorflow from data
        
        if self.model is None:
            self.model = self.__build_model(data_w_ret.shape, len(data.columns)) #creer le LSTM avec dim d'entrée, et en sortie le nb d'assets
        
        fit_predict_data = data_w_ret[np.newaxis,:]  # Add a dimension : for one year & 10 assets dimension of enter data is : (1,252,20)

        self.model.fit(fit_predict_data,                 # The training data no use
                       np.zeros((1, len(data.columns))), # The target data
                       epochs=20,                        # Nb of iterations to train
                       shuffle=False)                    # Boolean (whether to shuffle the training data before each epoch) 

        return self.model.predict(fit_predict_data)[0]   # Compute allocation weight : prediction of weights with training dataset