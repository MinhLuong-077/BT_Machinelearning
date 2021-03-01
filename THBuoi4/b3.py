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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_trans = sc_X.fit_transform(X)
Y_trans = sc_y.fit_transform(Y)
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_trans, Y_trans)
def predict(model, X, SC_X, SC_Y):
    X_trans = SC_X.transform(X)
    Y_trans_pred = model.predict(X_trans)
    Y_pred = SC_Y.inverse_transform(Y_trans_pred)
    return Y_pred
Y_pred_train = predict(regressor, X, sc_X, sc_y)
Y_pred_test = predict(regressor, X_test, sc_X, sc_y)
plt.scatter(X_test, Y_test, color = 'red')
plt.scatter(X_test, Y_pred_test, color = 'black')
plt.plot(X, Y_pred_train, color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(0, 11, 0.1)
X_grid= X_grid.reshape((len(X_grid), 1))
Y_pred_grid = predict(regressor, X_grid, sc_X, sc_y)
plt.scatter(X_test, Y_test, color = 'red')
plt.scatter(X_test,  Y_pred_test, color = 'black')
plt.plot(X_grid, Y_pred_grid, color = 'blue')
plt.title('Truth or Bluff (SVR)')
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
