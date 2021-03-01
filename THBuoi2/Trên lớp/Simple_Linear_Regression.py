import pandas as pd  #cho du lieu tu file
import numpy as np   #xu li mang 
import matplotlib.pyplot as plt #truc quan hóa dữ liệu
from sklearn.model_selection import train_test_split #phân chia dữ liệu
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#Tiền xử lí dữ liệu
dataset =pd.read_csv("Salary_Data.csv")
X=np.array(dataset.iloc[:,:-1].values)
Y=np.array(dataset.iloc[:,1].values)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,shuffle=True ,train_size=0.8, random_state=0)
reg=LinearRegression()
reg.fit(X_train, Y_train)
Y_train_pred=reg.predict(X_train)
Y_test_pred=reg.predict(X_test)
#hoàn thiện
plt.scatter(X_test, Y_test, color = 'red')
plt.scatter(X_test, Y_test_pred, color="BLACK")
plt.plot(X_train,Y_train_pred, color="BLUE")
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, Y_test, color = 'red')
plt.scatter(X_test, Y_test_pred, color="BLACK")
plt.plot(X_test,Y_test_pred, color="BLUE")
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
def compare(i_example):
    x=X_test[i_example : i_example+1]
    y=Y_test[i_example]
    y_pred=reg.predict(x)
    print(x,y,y_pred)
for i in range(len(X_test)):
    compare(i)
	#Đánh giá mô hình 
print('(R)2 train = ',reg.score(X_train, Y_train))
print('(R)2 test = ',reg.score(X_test, Y_test))
r2=r2_score(Y_test,Y_test_pred)
print('(R)2 test = ',r2)
if ((reg.score(X_train, Y_train)>=0.8)&(reg.score(X_test, Y_test)>=0.8)):
    print('Mô hình tốt')
elif ((reg.score(X_train, Y_train)==1)&(reg.score(X_test, Y_test)==1)):
     print('Mô hình cơ sở')
else:
    print('Cần xem lại')



