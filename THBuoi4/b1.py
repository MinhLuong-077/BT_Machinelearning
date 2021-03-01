import matplotlib.pyplot as plt
import pandas as pd

# dataset = pd.read_csv('Position_Salaries.csv')
dataset_train = pd.read_csv('Position_SalariesTrain.csv')
dataset_test = pd.read_csv('Position_SalariesTest.csv')
X = dataset_train.iloc[:, 1:-1].values
Y = dataset_train.iloc[:, -1].values
X_test = dataset_test.iloc[:, 1:-1].values
Y_test = dataset_test.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
Y_pred = lin_reg.predict(X)
Y_pred_test = lin_reg.predict(X_test)
plt.scatter(X_test, Y_test, color = "red")
plt.scatter(X_test, Y_pred_test, color = "black")
plt.plot(X, Y_pred, color = "blue")
plt.title("Position Level vs Salary (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary (dollars/year)")
plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt
print("SSE",len(X_test)*mean_squared_error(Y_test, Y_pred_test))
print("RMSE", sqrt(mean_squared_error(Y_test, Y_pred_test)))
from sklearn.metrics import r2_score
r2=r2_score(Y_test, Y_pred_test)
print("r2= ",r2)
adjusted_r_squared = 1 - (1-r2)*((len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1))
print("adjusted_r_squared",adjusted_r_squared)
import statsmodels.regression.linear_model as sm
regressor_OLS = sm.OLS(endog = Y, exog = X).fit() 
print(regressor_OLS.summary())
