# ML Model Housing Price Prediction
I recently started learning about machine learning models and I wanted to create a simple starter project using housing data from a Kaggle dataset.  I'm storing it here for future reference and to help teach others about the basics of machine learning. Hopefully at some point I will be able to show off a more intermediate or advanced level of model as I slowly build up this repository. 

## More About Me
I am beginner to the python language and I want to learn its utility for the purposes of analysing data.

## Dataset 
I retrieved this dataset from kaggle.com they site where I started learn about Machine learning. It is a great resource if you are just starting out like myself. 
https://www.kaggle.com/thomasnibb/amsterdam-house-price-prediction

## base model.py 
This is the very first ML model I was able to run. It is very simple model that uses the columns: Area (size of housing units), Room (number of rooms in unit), Lon and Lat (Longitude and Latitude, you can think of this as location). Having only 3 features also limits the predictive power of this model but the goal wasn't to make an imbecable model just yet. 

### Problems with this model
A key component of writing a good ML model is performance evaluation. So that we can understand how each iteration of our model affects its performance. From base model.ipynb, you can see that there is the line: 'mean_absolute_error(y, predict_base_dt_model)' which measures the mean absolute error between the predicted target (predict_base_dt_model) and the actual prices (y). 

However, the way this error is calculated is somewhat impractical. We want to test the accuracy of this model against NEW data, which is the main goal for these types of models we want to predict the price of a housing unit against housing units not already in the training data. Therefore, we should be calculating this error between two seperate groups of data, which is addressed in "randm_forest_and_xgboost_model.py".

## randm_forest_and_xgboost_model.py
A lot has changed since "base model .py", there are now 2 new ML models using RandomForestRegressor and XGBRegressor. Both of which are an improvement over the base model which uses the DecisionTreeRegressor. Notice how the mean absolute error here is much larger then the one seen in base model .ipynb. The reason for this large difference is because of the addition of the line: "X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)" which divides the dataset into a testing group (X_train and y_train) and a validation group (X_valid and y_valid). This addresses the problem stated earlier. Larger on I will be incorporating cross validation as anothe measure of evaluating the performance of the models. Given that the dataset is rather small, cross validation will limit the variation in our error calculation due to the random sampling of train_test_split. If you removed the random state variable and ran this file multiple time you will see how much the mean absolute error numbers vary. 

For the next step, I will be considering how we can handle missing values in the dataset, so far I have simply been dropping rows with missing data. Since our current dataset only drops 4 rows using another method like imputing wouldn't yield much of an improvement. So I will be using a different dataset with more columns and missing values. 

## XGBoost ML Model for Competition Submission
##### (Refer to XGboost_ML_model.py)
Currently this was used to create my first submission for this competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview.

Majority of the code used can be referenced from: https://www.kaggle.com/ryanholbrook/feature-engineering-for-house-prices/notebook

I have only included simple feature engineering techniques, i.e. mathematical transformations and group transformations. I will be looking to incorporate other methods like Principal Component Analysis and such. 
