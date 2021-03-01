import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values
Y = Y.reshape(len(Y),1)

#Polynomial_regression
from sklearn.preprocessing import PolynomialFeatures
poly_transform = PolynomialFeatures(degree=4)
X_poly = poly_transform.fit_transform(X)

from sklearn.linear_model import LinearRegression
poly_lin_reg = LinearRegression()
poly_lin_reg.fit(X_poly, Y)

Y_poly_pred = poly_lin_reg.predict(X_poly)
plt.scatter(X, Y, color = "red")
plt.plot(X, Y_poly_pred, color = "blue")
plt.title("Position Level vs Salary")
plt.xlabel("Position Level")
plt.ylabel("Salary (dollars/year)")
plt.show()

X_dummy = np.arange(0, 11, 0.1).reshape(-1, 1)
X_dummy_poly = poly_transform.transform(X_dummy)
Y_dummy_poly_pred = poly_lin_reg.predict(X_dummy_poly)
plt.scatter(X, Y, color = "red")
plt.scatter(X, Y, color = "red")
plt.plot(X_dummy, Y_dummy_poly_pred, color = "blue")
plt.title("Position Level vs Salary")
plt.xlabel("Position Level")
plt.ylabel("Salary (dollars/year)")
plt.show()

#Support_vector_regression
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_y.fit_transform(Y)
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(Y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_dummy = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)+5), 0.1)
X_dummy = X_dummy.reshape((len(X_dummy), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(Y), color = 'red')
plt.plot(X_dummy, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_dummy))), color = 'blue')
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(sc_X.transform(X))), color = "yellow")
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Decision_Tree_Regressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, Y)

X_grid = np.arange(min(X), max(X)+1, 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Random_Forest_Regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, Y)

X_grid = np.arange(min(X), max(X)+1, 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

