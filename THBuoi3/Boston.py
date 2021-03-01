import pandas as pd
import numpy as np
import time
import itertools
from sklearn.linear_model import LinearRegression
star = time.time()
# Importing the dataset
dataset = pd.read_csv('BostonTrain.csv')
dataset1 = pd.read_csv('BostonTest.csv')
Y_train=np.array(dataset.iloc[:, -1].values)
Y_test=dataset1.iloc[:,-1].values
K=[0,1,2,3,4,5,6,7,8,9,10,11,12]
re=[]
fd=[]
for i in itertools.combinations(K, 5):
    X_test=dataset1.iloc[:,[i[0],i[1],i[2],i[3],i[4]]].values
    X_train=dataset.iloc[:,[i[0],i[1],i[2],i[3],i[4]]].values
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, Y_train)
    lin_reg.score(X_train, Y_train)
    lin_reg.score(X_test, Y_test)
    re.append((lin_reg.score(X_test, Y_test)))
    fd.append(i)
    
dfs = pd.DataFrame(fd)
list_f=list(dataset.columns)
del list_f[-1]
dfs=dfs.replace(K,list_f)
dfs.insert(5,'R2',re)
dfs.to_csv("result10.csv",index=False)
print("Tổng thời gian chạy chương trình là = ", time.time()-star)