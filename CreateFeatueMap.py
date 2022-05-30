# -*- coding: utf-8 -*-
"""
Created on Thu May 26 12:41:29 2022

@author: qwe41
"""

import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler

# data = [_ for _ in range(100)]
# window = [0 for _ in range(5)]
# for i in range(100):
#     if i%5 == 0:
#         window = [0 for _ in range(5)]
#     window[i%5] = data[i]
#     print(i,":",window)

#%%

df = pd.read_csv('D:/code/data/FuturesData15M.csv')
df.index = pd.to_datetime(df['datetime'])
# w1 = df.loc['2017-05-17 08:45':'2017-05-24 13:30']
# w2 = df.loc['2017-05-24 08:45':'2017-05-31 13:30']
# w3 = df.loc['2017-05-31 08:45':'2017-06-07 13:30']
# w4 = df.loc['2017-06-07 08:45':'2017-06-14 13:30']


st = '2017-05-17 08:45:00'
ed = '2017-05-24 13:30:00'
featuremap = []
while  datetime.datetime.strptime(str(ed),'%Y-%m-%d %H:%M:%S') <= datetime.datetime.strptime("2022-04-14",'%Y-%m-%d'):
    print(st,ed)
    data = df.loc[st:ed]
    features = [[0,0,0,0,0] for _ in range(len(data))]
    for i in range(len(data)):
       features[i]  = data.iloc[i,1:6].to_numpy(dtype=np.float32)
       featuremap.append(features)
    st = datetime.datetime.strptime(str(st),'%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=7)
    ed = datetime.datetime.strptime(str(ed),'%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=7)


featuremap_array = np.array(featuremap)
scaler =MinMaxScaler()
featuremap_array = scaler.fit_transform(featuremap_array)   
np.save('FeatureMap',featuremap_array)