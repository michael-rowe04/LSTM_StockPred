import yfinance as yf
import pandas as pd
import pandas_ta as ta
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,normalize
import numpy as np


class dataset:
    def __init__(self,ticker,period,indicators_keyword): #Tell it ticker symbol, period you want to check, and function of indicators want to check
        self.ticker = ticker
        self.period = period
        self.indicators_keyword = indicators_keyword
        stock = yf.Ticker(self.ticker)
        self.hist = stock.history(self.period)
        #self.ds = pd.DataFrame() #I feel like I just keep remaking the ds, maybe try self.ds = self.make_ds(), that way it makes it off jump we can just use self.ds
        #my above comment worked
        self.ds = self.make_ds()

    def make_ds(self):
        self.ds = pd.DataFrame()
        if self.indicators_keyword == 'OG':
            self.ds = self.ds.join(self.og_indicators())
        
        self.ds = self.ds.dropna()
        #self.ds = self.ds/self.ds.max()
        cols = list(self.ds)[1:]
        self.ds[cols] = self.ds[cols]/self.ds[cols].max() #this transforms everything except the percent change now
        #might need to make this all float vals, not sure, also scikit learn preproccessing drops headers and date so need to look into that
        #self.ds = MinMaxScaler(feature_range = (-1,1)).fit_transform(self.ds)
        #self.ds = pd.DataFrame(self.ds)
        return self.ds
    
    def split(self,train_percent):
        #These two do same thing
        train_size = (int)(train_percent * len(self.make_ds()))
        todays_data = self.ds.iloc[-1:,1:]
        x = self.ds.iloc[:,1:]
        y = self.ds.iloc[:,0].shift(-1)
        X_train = x[:train_size]
        y_train = y[:train_size]
        X_test = x[train_size:]
        y_test = y[train_size:]
        #X_train,X_test,y_train,y_test = train_test_split(self.ds.iloc[:,1:], self.ds.iloc[:,0], train_size = train_percent, shuffle = False) 
        #Getting overlapping indices from this because of join function, need to find a better method to combined dataframes
        #Wasnt actually from join function, it was from me continually calling make_ds()
        #Not going to use scikit learn split because I need the y values to be one day ahead of test values so that it is actually predicting the future
        return X_train,X_test,y_train,y_test,todays_data
    
    def moving_window(self,n_future,n_past):
        #I think this is how lstms are supposed to be used, this inputs multiple timesteps each input
        #have to change the input size in model to (num_samples,num_features)
        trainX = []
        trainY = []
        np_df = self.ds.to_numpy()

        #n_future   # Number of days we want to look into the future based on the past days.
        #n_past  # Number of past days we want to use to predict the future.

        #Reformat input data into a shape: (n_samples x timesteps x n_features)
        for i in range(n_past, len(np_df) - n_future +1):
            trainX.append(np_df[i - n_past:i, 1:np_df.shape[1]])
            trainY.append(np_df[i + n_future - 1:i + n_future, 0])

        trainX, trainY = np.array(trainX), np.array(trainY)


    def og_indicators(self): #Have to make a new one of these functions when you want to test different indicators
        #self.ds['Close'] = self.hist['Close']/self.hist['Close'].max() #Every time you make a new one you will need to check out what the indicators return

        self.ds['Change'] = self.hist['Close'].pct_change()
        self.ds['RSI'] = ta.rsi(self.hist['Close'],length =14) #14 nas, returns with no column label so need to do it this way
        #self.ds['RSI'] = rsi/rsi.max()
        aroon = ta.aroon(self.hist['High'],self.hist['Low']) #14 nas
        #aroon = aroon/aroon.max()
        macd = ta.macd(self.hist['Close'])#33 nas, MACD 25 nas, MACDh & MACDs 33 nas
        #macd = macd/macd.max()
        adx = ta.adx(self.hist['High'],self.hist['Low'],self.hist['Close']) #this might uses data straight from the day we are trying to predict
        #adx = adx/adx.max()
        return [aroon,macd,adx]
    
    """ def og_indicators_test(self):
        self.ds['Close'] = self.hist['Close'] #Every time you make a new one you will need to check out what the indicators return
        self.ds['RSI'] = ta.rsi(self.hist['Open'],length =14) #14 nas, returns with no column label so need to do it this way
        aroon = ta.aroon(self.hist['High'],self.hist['Low']) #14 nas
        macd = ta.macd(self.hist['Close'])#33 nas, MACD 25 nas, MACDh & MACDs 33 nas
        #adx = ta.adx(self.hist['High'],self.hist['Low'],self.hist['Close']) this might uses data straight from the day we are trying to predict
        return [aroon,macd] """


""" stock = yf.Ticker('ASML')
hist = stock.history('6mo')

ds = pd.DataFrame()
ds['RSI'] = ta.rsi(hist['Open'],length =14)

ds.dropna()
ds.drop(ds.tail(1).index, inplace=True)
df = pd.DataFrame()
df['Close'] = hist['Close']
df.drop(df.head(1).index, inplace=True)
da = pd.DataFrame()
da = ds.join(df)
norm = MinMaxScaler(feature_range = (0,1))

da_scaled = norm.fit_transform(da)
da_scaled_df = pd.DataFrame(da_scaled)
da_scaled
da
new = pd.DataFrame()
hist['Close']
prev_close = hist['Close'].shift(1)
new['Close'] = hist['Close']
new['prev Close'] = prev_close
new
new["prevClose"] = hist['Close'].shift(-1)
new['change'] = new[['Close','prev Close']].pct_change(axis=1)['Close']
new['change'] = new['Close'].pct_change()
new['change'] = hist['Close'].pct_change()
new_scaled = norm.fit_transform(new)
new_norm = normalize(new.dropna(),'max')
new.max()
new.dropna()
.008964/new.max()
new.max().max()
new['Close'] = new['Close']/new['Close'].max()
new = new/new.max()
new """