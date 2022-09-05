# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:40:13 2022

@author: sp7012
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras as keras
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler


import pandas as pd

TrL=1100
ARO=120
df = pd.read_csv('HD.csv')

df = df[['Date', 'Close']]
df.head()

df = df.replace({'\$':''}, regex = True)

df = df.astype({"Close": float})
df["Date"] = pd.to_datetime(df.Date, format="%m/%d/%Y")
df.dtypes

df.index = df['Date']

plt.plot(df["Close"],label='Close Price history')
df = df.sort_index(ascending=True,axis=0)
data = pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])
for i in range(0,len(data)):
    data["Date"][i]=df["Date"][i]
    data["Close"][i]=df["Close"][i]
data.head()

scaler=MinMaxScaler(feature_range=(0,1))
data.index=data.Date
data.drop("Date",axis=1,inplace=True)
final_data = data.values
train_data=final_data[0:TrL,:]
valid_data=final_data[TrL:,:]
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_data)
x_train_data,y_train_data=[],[]
for i in range(ARO,len(train_data)):
    x_train_data.append(scaled_data[i-ARO:i,0])
    y_train_data.append(scaled_data[i,0])
    
    
lstm_model=Sequential()
lstm_model.add(LSTM(units=90,return_sequences=True,input_shape=(np.shape(x_train_data)[1],1)))
lstm_model.add(LSTM(units=90))
lstm_model.add(Dense(1))
model_data=data[len(data)-len(valid_data)-ARO:].values
model_data=model_data.reshape(-1,1)
model_data=scaler.transform(model_data)

x_train_data = np.asarray(x_train_data)
y_train_data = np.asarray(y_train_data)

lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)
X_test=[]
for i in range(ARO,model_data.shape[0]):
    X_test.append(model_data[i-ARO:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))



predicted_stock_price=lstm_model.predict(X_test)
predicted_stock_price=scaler.inverse_transform(predicted_stock_price)


train_data=data[:TrL]
valid_data=data[TrL:]
valid_data['Predictions']=predicted_stock_price
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close',"Predictions"]])
