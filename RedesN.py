# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:32:05 2022

@author: Angel Eduardo Zecua Hdz
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')
df['Fecha']=pd.to_datetime(df['Timestamp'],unit='s').dt.date
date=df.groupby('Fecha')
data=fechas['Close'].mean()

len(data)
x_tr=data.iloc[:len(data)-50] 
x_te=data.iloc[len(x_tr):] 

x_tr=np.array(x_tr)
x_tr=x_tr.reshape(x_tr.shape[0],1)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))
xtr_scaled=scaler.fit_transform(x_tr)

pasos=50
x_tr=[]
y_tr=[]

for i in range(pasos,xtr_scaled.shape[0]):
    x_tr.append(xtr_scaled[i-pasos:i,0])
    y_tr.append(xtr_scaled[i,0])
    
len(x_tr)

x_tr=np.array(x_tr)

y_tr=np.array(y_tr)

x_tr=x_tr.reshape(x_tr.shape[0],x_tr.shape[1],1)


model=Sequential()
   
 
model.add(LSTM(254,input_shape=(None,1), activation = 'relu'))
model.add(Dropout(0.2))
    
model.add(Dense(1,activation = 'linear'))
    
model.compile(loss="mean_squared_error",optimizer="adam")

model.summary()

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(x_tr, y_tr,
                    batch_size=64,
                    epochs=20,
                    verbose=1)
