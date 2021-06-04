import pandas as pd
import numpy as np
import pickle

# Read the Dataset
df = pd.read_csv("DriverBehaviourfinal.csv")

# Summarize shape
df.shape

# Summarize first few lines
df.head()

# Get the Independent and Dependent Features
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Parameters to be tuned 
params = {
 "learning_rate"    : [ 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
 "subsample"        : [0.5, 0.7],
 "n_estimators"     : [100, 200, 500, 1000]
}

import xgboost
regressor = xgboost.XGBRegressor(objective = 'reg:squarederror')

random_search = RandomizedSearchCV(regressor, param_distributions=params, n_iter=5, scoring = 'neg_mean_squared_error', n_jobs = -1, cv = 5, verbose = 1)

random_search.fit(X, y)

random_search.best_estimator_

random_search.best_params_

regressor = xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.7, gamma=0.0,
             importance_type='gain', learning_rate=0.05, max_delta_step=0,
             max_depth=8, min_child_weight=5, missing=None, n_estimators=500,
             n_jobs=1, nthread=None, objective='reg:squarederror',
             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
             seed=None, silent=None, subsample=0.7, verbosity=1)

regressor.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

# Define model evaluation method
cv = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 1)

# Evaluate model
scores = cross_val_score(regressor, X, y, scoring = 'neg_mean_squared_error', cv = cv, n_jobs = -1)

# Force scores to be positive
scores = np.absolute(scores)
print('Mean MSE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

# Predict the output
y_test_pred = regressor.predict(X_test)

# Compute performance metrics
import sklearn.metrics as sm
from math import sqrt
print("XGB regressor performance:\n")
mse = sm.mean_squared_error(y_test, y_test_pred)
print("Mean squared error =", round(mse, 3))
rmse = sqrt(mse)
print("RMSE = ",round(rmse, 3))
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 3)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 3)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 3))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 3))

pickle.dump(regressor,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

data = X_test[0].reshape(1, -1)
print(data)
p = regressor.predict(data)
print( "Prediction of Model: ", p, "Actual result:", y_test[0])

data2 = np.array([35, 1, 0, 1, 0, 5, 0, 1]).reshape(1, -1)
print(data2)
p2 = regressor.predict(data2)
print("Prediction2 of Model : ", p2)

data3 = np.array([ 45, 1, 0, 1, 0, 8, 0, 0 ]).reshape(1, -1)
print(data3)
p3 = regressor.predict(data3)
print("Prediction3 of Model : ", p3)