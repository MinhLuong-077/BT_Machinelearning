import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# dataset = pd.read_csv('Position_Salaries.csv')
dataset_train = pd.read_csv('city_day_AQI_train.csv')
dataset_test = pd.read_csv('city_day_AQI_test.csv')
del dataset_train['Date']
del dataset_test['Date']
dataset_train.fillna(dataset_train.groupby('City').transform('mean'),inplace=True)
dataset_test.fillna(dataset_test.groupby('City').transform('mean'),inplace=True)
dataset_train = dataset_train.fillna(dataset_train.mean())
dataset_test = dataset_test.fillna(dataset_test.mean())
X = dataset_train.iloc[:, :-2].values
Y = dataset_train.iloc[:, -2].values
X_test = dataset_test.iloc[:,:-2].values
Y_test = dataset_test.iloc[:, -2].values
a=X_test
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X_test = np.array(ct.transform(X_test))
Y = Y.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 7,random_state=0  )
regressor.fit(X, Y)
Y_pred_test=regressor.predict(X_test)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
print("SSE",len(X_test)*mean_squared_error(Y_test,Y_pred_test))
print("RMSE", sqrt(mean_squared_error(Y_test, Y_pred_test)))
r2 = r2_score(Y_test, Y_pred_test)
print("r2=",r2)
adjusted_r_squared = 1 - (1-r2)*((len(Y_test)-1)/(len(Y_test)-a.shape[1]-1))
print("adjusted_r_squared= ",adjusted_r_squared)



