# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 19:36:22 2020

@author: TTC
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# dataset = pd.read_csv('Position_Salaries.csv')
dataset_train = pd.read_csv('Position_SalariesTrain.csv')
dataset_test = pd.read_csv('Position_SalariesTest.csv')
X = dataset_train.iloc[:, 1:-1].values
Y = dataset_train.iloc[:, -1].values
X_test = dataset_test.iloc[:, 1:-1].values
Y_test = dataset_test.iloc[:, -1].values
Y = Y.reshape(len(Y),1)
Y_test = Y_test.reshape(len(Y_test),1)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X, Y)
Y_pred_test=regressor.predict(X_test)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, Y_test, color = 'red')
plt.scatter(X_test, Y_pred_test, color = 'black')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt
print("SSE",len(X_test)*mean_squared_error(Y_test, Y_pred_test))
print("RMSE", sqrt(mean_squared_error(Y_test, Y_pred_test)))
from sklearn.metrics import r2_score
r2=r2_score(Y_test, Y_pred_test)
print("r2=",r2)
adjusted_r_squared = 1 - (1-r2)*((len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1))
print("adjusted_r_squared= ",adjusted_r_squared)