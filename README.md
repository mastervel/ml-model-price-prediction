# ML Model Housing Price Prediction
I recently started learning about machine learning models and I wanted to create a simple starter project using housing data from a Kaggle dataset.  I'm storing it here for future reference and to help teach others about the basics of machine learning. Hopefully at some point I will be able to show off a more intermediate or advanced level of model as I slowly build up this repository. 

## More About Me
I am beginner to the python language and I want to learn its utility for the purposes of analysing data.

## Dataset 
I retrieved this dataset from kaggle.com they site where I started learn about Machine learning. It is a great resource if you are just starting out like myself. 
https://www.kaggle.com/thomasnibb/amsterdam-house-price-prediction

## Base Model (see:base model.py) 
This is the very first ML model I was able to run. It is very simple model that uses the columns: Area (size of housing units), Room (number of rooms in unit), Lon and Lat (Longitude and Latitude, you can think of this as location). Having only 3 features also limits the predictive power of this model but the goal wasn't to make an imbecable model just yet. 

### Problems with this model
A kay part of any ML model, I learned, is that you need to find a way to properly measure its predictive power. So that we can understand how each iteration of our model improves or worsens its predictive power. From base model.ipynb, you can see that there is the line: 'mean_absolute_error(y, predict_base_dt_model)' which measures the mean absolute error between the predicted target (predict_base_dt_model) and the actual prices (y). 

However, the way this error is calculated is somewhat impractical. We want to test the accuracy of this model against NEW data, which is the main goal for these types of models we want to predict the price of a housing unit against housing units not already in the training data. Therefore, we should be calculating this error between two seperate groups of data. The first group which we will use to train the model called the training data and the second group of data used to test the validity of the model, therefore we shall call this the validation data. We will do this in the next model. 
