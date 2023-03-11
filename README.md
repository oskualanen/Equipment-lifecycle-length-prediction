# my-new-project
Building AI course project

# This is an idea to use the K nearest neighbour algorithm to predict the optimal lifecycle length of a laboratory equipment. In this model, Y is the age of the instrument at the point when it is disposed. There are a total of values of X (X1-X4) consisting of parameters that are most likely to affecting the optimal length of the instrument. These may include: (1) the number of failure events, (2) the utilization hours of the instrument, (3) the years left of support from the instrument provider and (4) the number of backup instruments in the laboratory.

## This is KKN model
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
