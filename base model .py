# Setup
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# importing the data from the csv
filepath = '/Users/veliristimaki/Code/ML model for housing price/HousingPrices-Amsterdam-August-2021.csv'
house_data = pd.read_csv(filepath)

# removing rows with missing data and dropping useless column
house_data = house_data.drop('Unnamed: 0', 1)
house_data = house_data.dropna(axis=0)

base_dt_model = DecisionTreeRegressor()

# creating a variable y that is specifying the target (what we want to predict)
y = house_data.Price
# creating a variable X that are the features of the model, i.e. what will be used to predict price
features = ['Area', 'Room', 'Lon', 'Lat']
X = house_data[features]

# fitting our model with X and y:
base_dt_model.fit(X,y)
predict_base_dt_model = base_dt_model.predict(X)

# checking the accuracy of our model
mean_absolute_error(y, predict_base_dt_model)



