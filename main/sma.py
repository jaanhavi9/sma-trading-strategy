#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
import plotly.express as px
import matplotlib.pyplot as plt
import unittest
plt.style.use('fivethirtyeight')


# In[2]:


#Inserting data from excel to MySQL database


# In[3]:


df = pd.read_excel('HINDALCO_1D.xlsx')


# In[4]:


engine = create_engine('mysql://root:Janhavi@123@localhost/hindalco')


# In[5]:


df.to_sql('hindalco2', con = engine)


# In[ ]:


df = df.set_index(pd.DatetimeIndex(df['datetime'].values))


# In[ ]:


#30 day simple moving average trading strategy


# In[7]:


#closing price plot 
plt.figure(figsize = (16,8))
plt.title('Closing Price')
plt.plot(df['close'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()


# In[22]:


def SMA(data, period = 30, column = 'close'):
    return data[column].rolling(window = period).mean()


# In[82]:


df['SMA30'] = SMA(df)
print(df)


# In[24]:


def strategy(df):
    buy = []
    sell = []
    flag = 0
    buy_price = 0 
    
    for i in range(0, len(df)):
        if df['SMA30'][i]  >  df['close'][i]  and flag == 0:
            buy.append(df['close'][i])
            sell.append(np.nan)
            buy_price = df['close'][i]
            flag = 1
        elif df['SMA30'][i]  <  df['close'][i]  and flag == 1 and buy_price < df['close'][i]:
            sell.append(df['close'][i])
            buy.append(np.nan)
            buy_price = 0
            flag = 0
        else:
            sell.append(np.nan)
            buy.append(np.nan)
    
    return(buy,sell)
            


# In[25]:


strat = strategy(df)
df['buy'] = strat[0]
df['sell'] = strat[1]


# In[26]:


plt.figure(figsize = (16,8))
plt.title('Closing Price with BUY and SELL signal')
plt.plot(df['close'],  alpha = 0.5, label = 'close')
plt.plot(df['SMA30'], alpha = 0.5, label = 'SMA30')
plt.scatter(df.index, df['buy'], color = 'green', label = 'Buy Signal', marker = '^', alpha = 1)
plt.scatter(df.index, df['sell'], color = 'red', label = 'Sell Signal', marker = 'v', alpha = 1)
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()


# In[53]:


#unit testing


# In[78]:


#check if instrument is string
def is_string():
    for i in df['instrument']:
        if type(i) == str:
            print("true")
        else:
            print("False")


# In[70]:


#check if volume is integer
def is_int():
    for i in df['volume']:
        if type(i) == int:
            print("True")
        else:
            print("False")


# In[80]:


#check if open close high low is float
def is_float():
    for i in df['open']:
        if type(i) == float:
            print('True')
        else:
            print("False")
    for i in df['close']:
        if type(i) == float:
            print('True')
        else:
            print("False")
    for i in df['high']:
        if type(i) == float:
            print('True')
        else:
            print("False")
    for i in df['low']:
        if type(i) == float:
            print('True')
        else:
            print("False")


# In[81]:


class test_input(unittest.TestCase):
    def test_string(self):
        self.assertTrue(is_string())
    def test_int(self):
        self.assertTrue(is_int())
    def test_float(self):
        self.assertTrue(is_float())

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[ ]:




