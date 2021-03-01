import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
# dataset= pd.read_csv("50_Startups.csv")
# X = dataset.iloc[:,0:4].values
# Y = dataset.iloc[:,-1].values
dataset= pd.read_csv("50_Startups_Train.csv")
dataset1= pd.read_csv("50_Startups_Test.csv")
X_train = dataset.iloc[:,1:5].values
Y_train= dataset.iloc[:,-1].values

X_test = dataset1.iloc[:,1:5].values
Y_test = dataset1.iloc[:,-1].values

# le = LabelEncoder()
# le.fit(X[:,3])
# X[:,3]= le.fit_transform(X[:,3])
# ohe = OneHotEncoder(handle_unknown = 'ignore')
# df = ohe.fit_transform(dataset[['State']]).toarray()
# X = np.concatenate((df, X), axis = 1)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X_train = np.array(ct.fit_transform(X_train))
X_test = np.array(ct.fit_transform(X_test))

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
print("R(2)_Train= ",lin_reg.score(X_train, Y_train))
print("R(2)_Test= ",lin_reg.score(X_test, Y_test))
def compare(i_example):
     x = X_test[i_example : i_example + 1]
     y = Y_test[i_example]
     y_pred = lin_reg.predict(x)[0]
     print(x[:,3:6],x[:, 0:3],y, y_pred)
for i in range(len(X_test)):
    compare(i)
