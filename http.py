# http.py

import json
from nameko.web.handlers import http
#import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
#import matplotlib.pyplot as plt
from keras.models import load_model
import pickle

class HttpService:
    name = "http_service"

    @http('GET', '/predictValue/<int:value>')
    def predictValue(self, request, value):
        file = open('model/scaler.pkl', 'rb')
        # dump information to that file
        scaler = pickle.load(file)
        # close the file
        file.close()
        
        apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
        #Create a new dataframe
        new_df = apple_quote.filter(['Close'])
        #Get teh last 60 day closing price 
        last_60_days = new_df[-60:].values
        #Scale the data to be values between 0 and 1
        last_60_days_scaled = scaler.transform(last_60_days)
        #Create an empty list
        X_test = []
        #Append teh past 60 days
        X_test.append(last_60_days_scaled)
        #Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        #Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
         #Get the predicted scaled price
        pred_price = model.predict(X_test)
        print(pred_price) 
        #undo the scaling 
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)           
    
    
    
        return json.dumps({'value': value})

# so we need 2 fonctions  the fist predictValue that predict one value  by giving time as argument
#the second will gill us the full predicted value of the dates given
