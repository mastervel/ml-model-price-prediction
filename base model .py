#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
print('Setup Complete')


# In[2]:


filepath = '/Users/veliristimaki/Code/ML model for housing price/HousingPrices-Amsterdam-August-2021.csv'
house_data = pd.read_csv(filepath)
house_data


# In[3]:


house_data = house_data.drop('Unnamed: 0', 1)
house_data = house_data.dropna(axis=0)


# In[4]:


house_data


# In[5]:


base_dt_model = DecisionTreeRegressor()
y = house_data.Price
features = ['Area', 'Room', 'Lon', 'Lat']
X = house_data[features]
base_dt_model.fit(X,y)
predict_base_dt_model = base_dt_model.predict(X)
mean_absolute_error(y, predict_base_dt_model)


# In[ ]:




