# Using machine learning to predict the optimal lifecycle length of equipment.

Building AI course project

## Summary

This is an idea to use the K nearest neighbour algorithm to predict the optimal lifecycle length of an equipment. In this model, Y is the age of the instrument at the point when it is disposed. There are a total of values of X (X1-X4) consisting of parameters that are most likely to affecting the optimal lifecycle length of the equipment.

## Background

Evaluating the best lifecycle length of equipment is an important task in a business setting, because purchasing new equipment costs money (from tens of thousands of euros to millions of euros). As such, it is resonable to expect that the equipment should be utilized in a production settings for as long as possible.

## This is the proposed KKN model:

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Read the data from the csv file and clean it up
data = pd.read_csv("data.csv")
data = data.replace({",": "."}, regex=True)
data = data[['X_1', 'X_2', 'X_3', 'X_4', 'Y']].astype(float)

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)

# Define the feature columns and target column
X_train = train_data[['X_1','X_2','X_3','X_4']].values
y_train = train_data["Y"].values

X_test = test_data[['X_1','X_2','X_3','X_4']].values
y_test = test_data["Y"].values

# Create a KNeighborsRegressor model
knn_reg = KNeighborsRegressor()

# Use GridSearchCV to find the best value for k
param_grid = {'n_neighbors':[1,3,5,7,9]}
grid_search = GridSearchCV(knn_reg, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Train the model on the training data
knn_reg = grid_search.best_estimator_
knn_reg.fit(X_train, y_train)

# Make predictions on the test set
predictions = knn_reg.predict(X_test)

# Evaluate the model's performance
print("Mean squared error:", mean_squared_error(y_test, predictions))
print("R-squared:", r2_score(y_test, predictions))


## Challenges

The optimal lifecycle length obtained with this model is only as good as the data used to train the model. As such, only accurate data should be used. Additionally, the used ML model is likely not the best one for lifecycle length prediction. Other ML algorithms like XGboost or random forest could be utilized as well. Again, this depends on the amount of data (and the quality.

## Acknowledgements 

This idea is an original idea, which attemps to utilize a KNN algorithm for predicting lifecycle length of equipment. 
