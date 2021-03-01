import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# dataset = pd.read_csv('Position_Salaries.csv')
dataset_train = pd.read_csv('city_day_AQI_train.csv')
dataset_test = pd.read_csv('city_day_AQI_test.csv')
del dataset_train['Date']
del dataset_test['Date']
dataset_train.fillna(dataset_train.groupby('City').transform('mean'))
dataset_test.fillna(dataset_test.groupby('City').transform('mean'))
dataset_train = dataset_train.fillna(dataset_train.mean())
dataset_test = dataset_test.fillna(dataset_test.mean())
X = dataset_train.iloc[:, :-2].values
Y = dataset_train.iloc[:, -2].values
X_test = dataset_test.iloc[:,:-2].values
Y_test = dataset_test.iloc[:, -2].values
a=X_test
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X_test = np.array(ct.fit_transform(X_test))

lin_reg = LinearRegression()
lin_reg.fit(X, Y)
Y_test_pred = lin_reg.predict(X_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
print(lin_reg.score(X_test,Y_test))
rmse = sqrt(mean_squared_error(Y_test, Y_test_pred))
mse = mean_squared_error(Y_test, Y_test_pred)
mae = mean_absolute_error(Y_test, Y_test_pred)
r2 = r2_score(Y_test, Y_test_pred)
adjusted_r_squared = 1 - (1-r2)*((len(Y_test)-1)/(len(Y_test)-a.shape[1]-1))
print("adjusted_r_squared= ",adjusted_r_squared)
print("SSE",len(X_test)*mean_squared_error(Y_test, Y_test_pred))
print("Score RSME:",rmse)
print("Score MSE:",mse)
print("Score MAE:",mae)
print("Score R2:",r2)


from sklearn.preprocessing import PolynomialFeatures
poly_transform = PolynomialFeatures(degree=2)
X_poly = poly_transform.fit_transform(X)
X_poly_test = poly_transform.transform(X_test)
from sklearn.linear_model import LinearRegression
poly_lin_reg = LinearRegression()
poly_lin_reg.fit(X_poly, Y)

Y_poly_pred_test = poly_lin_reg.predict(X_poly_test)
print("SSE",len(X_test)*mean_squared_error(Y_test, Y_poly_pred_test))
print("RMSE", sqrt(mean_squared_error(Y_test, Y_poly_pred_test)))
from sklearn.metrics import r2_score
r2=r2_score(Y_test, Y_poly_pred_test)
print("r2=",r2)
adjusted_r_squared = 1 - (1-r2)*((len(Y_test)-1)/(len(Y_test)-a.shape[1]-1))
print("adjusted_r_squared= ",adjusted_r_squared)

Y = Y.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X, Y)
Y_pred_test=regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
from math import sqrt
print("SSE",len(X_test)*mean_squared_error(Y_test, Y_pred_test))
print("RMSE", sqrt(mean_squared_error(Y_test, Y_pred_test)))
from sklearn.metrics import r2_score
r2=r2_score(Y_test, Y_pred_test)
print("r2=",r2)
adjusted_r_squared = 1 - (1-r2)*((len(Y_test)-1)/(len(Y_test)-a.shape[1]-1))
print("adjusted_r_squared= ",adjusted_r_squared)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 7,random_state=0  )
regressor.fit(X, Y)
Y_pred_test=regressor.predict(X_test)
from sklearn.metrics import mean_squared_error
from math import sqrt
print("SSE",len(X_test)*mean_squared_error(Y_test, Y_pred_test))
print("RMSE", sqrt(mean_squared_error(Y_test, Y_pred_test)))
from sklearn.metrics import r2_score
r2=r2_score(Y_test, Y_pred_test)
print("r2=",r2)
adjusted_r_squared = 1 - (1-r2)*((len(Y_test)-1)/(len(Y_test)-a.shape[1]-1))
print("adjusted_r_squared= ",adjusted_r_squared)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_trans = sc_X.fit_transform(X)
Y_trans = sc_y.fit_transform(Y)

X_trans_test = sc_X.transform(X_test)
Y_trans_test = sc_y.transform(Y_test)
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
rmse = sqrt(mean_squared_error(Y_test, Y_pred_test))
mse = mean_squared_error(Y_test, Y_pred_test)
mae = mean_absolute_error(Y_test, Y_pred_test)
r2 = r2_score(Y_test, Y_pred_test)
adjusted_r_squared = 1 - (1-r2)*((len(Y_test)-1)/(len(Y_test)-a.shape[1]-1))
print("adjusted_r_squared= ",adjusted_r_squared)
print("SSE",len(X_test)*mean_squared_error(Y_test, Y_pred_test))
print("Score RSME:",rmse)
print("Score MSE:",mse)
print("Score MAE:",mae)
print("Score R2:",r2)










