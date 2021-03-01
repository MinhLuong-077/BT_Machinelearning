import pandas as pd  #cho du lieu tu file
import numpy as np   #xu li mang 
import matplotlib.pyplot as plt #truc quan hóa dữ liệu
from sklearn.model_selection import train_test_split #phân chia dữ liệu
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#Tiền xử lí dữ liệu
dataset =pd.read_csv("Salary_Data.csv")
dataset1 =pd.read_csv("Salary_Data_Test.csv")
X=np.array(dataset.iloc[:,:-1].values)
Y=np.array(dataset.iloc[:,1].values)
X1_test=np.array(dataset1.iloc[:,:-1].values)
Y1_test=np.array(dataset1.iloc[:,1].values)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.5, random_state=0)
reg=LinearRegression()
reg.fit(X_train, Y_train)
Y_train_pred=reg.predict(X_train)
Y_test_pred=reg.predict(X1_test)
#hoàn thiện
plt.scatter(X1_test, Y1_test, color = 'red')
plt.scatter(X1_test, Y_test_pred, color="BLACK")
plt.plot(X_train,Y_train_pred, color="BLUE")
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
def compare(i_example):
       x=X1_test[i_example : i_example+1]
       y=Y1_test[i_example]
       y_pred=reg.predict(x)
       print(x,y,y_pred)
for i in range(len(X1_test)):
    compare(i)
#Đánh giá mô hình 
from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y1_test, Y_test_pred))
print('(R)2 test = ',reg.score(X1_test, Y1_test))
r2=r2_score(Y1_test,Y_test_pred)
print('(R)2 test = ',r2)
