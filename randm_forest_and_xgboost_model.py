#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')
print('Setup Complete')


# In[58]:


df = pd.read_csv('/Users/veliristimaki/Code/ML model for housing price/HousingPrices-Amsterdam-August-2021.csv')
df = df.drop('Unnamed: 0', 1)
df = df.dropna(axis=0)
features = ['Area', 'Room', 'Lon', 'Lat']
X = df[features]
y = df.Price
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)
print('Base Data Split Completed')


# In[59]:


xgb_model = XGBRegressor(random_state=1)
xgb_model.fit(X_train,y_train)
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(X_train,y_train)
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X_train,y_train)
print('Fitting Models Complete')


# In[60]:


xgb_predictions = xgb_model.predict(X_valid)
dt_predictions = dt_model.predict(X_valid)
rf_predictions = rf_model.predict(X_valid)
print("Mean Absolute Error of extreme gradient boosting model: " + str(mean_absolute_error(xgb_predictions, y_valid)))
print("Mean Absolute Error of Random Forest model: " + str(mean_absolute_error(rf_predictions, y_valid)))
print("Mean Absolute Error of Decision Tree Model: " + str(mean_absolute_error(dt_predictions, y_valid)))


# In[ ]:




