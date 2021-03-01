import pandas as pd 
import numpy as np 
import math
from numpy.linalg import norm
from scipy.spatial import distance
from scipy.spatial.distance import pdist

dataset =pd.read_csv("Point_Data.csv")
p=dataset.iloc[0,[0,1]]
q=dataset.iloc[1,[0,1]]
Eculid=norm(p-q)
print("Khoảng cách Eculid = ",Eculid)
K=[]
for i in dataset.itertuples():
    K.append((round(i.x,3),round(i.y,3)))
print("Số điểm: ",len(K))
re=[]
for j in K[:10]:
    print(j)
#Điểm 10
for i in range(10):
    x1,y1=K[i]
    row=[]
    p=np.array([x1,y1])
    for j in range(10):
        x2,y2=K[j]
        q=np.array([x2,y2])
        dist = norm(p-q)
        row.append(round(dist,3))
    re.append(row)

dfs = pd.DataFrame(re)
dfs.to_csv("resultEculid10.csv",index=False)

re1=[]
for i in range(10):
    x1,y1=K[i]
    row=[]
    p=np.array([x1,y1])
    for j in range(10):
        x2,y2=K[j]
        q=np.array([x2,y2])
        dist = distance.cityblock(p,q)
        row.append(round(dist,3))
    re1.append(row)

dfs = pd.DataFrame(re1)
dfs.to_csv("resultMahattan10.csv",index=False)

re2=[]
r=int(input("Nhap r:"))
for i in range(10):
    x1,y1=K[i]
    row=[]
    p=np.array([x1,y1])
    for j in range(10):
        x2,y2=K[j]
        q=np.array([x2,y2])
        dist = distance.minkowski(p,q,r)
        row.append(round(dist,3))
    re2.append(row)

dfs = pd.DataFrame(re2)
dfs.to_csv("resultminkowski10.csv",index=False)

re3=[]
for i in range(10):
    x1,y1=K[i]
    row=[]
    p=np.array([x1,y1])
    for j in range(10):
        x2,y2=K[j]
        q=np.array([x2,y2])
        dist = distance.chebyshev(p,q)
        row.append(round(dist,3))
    re3.append(row)

dfs = pd.DataFrame(re3)
dfs.to_csv("resultChessboard.csv",index=False)

re4=[]
for i in range(10):
    x1,y1=K[i]
    row=[]
    p=np.array([x1,y1])
    for j in range(10):
        x2,y2=K[j]
        q=np.array([x2,y2])
        dist = distance.hamming(p,q)
        row.append(round(dist,3))
    re4.append(row)

dfs = pd.DataFrame(re4)
dfs.to_csv("result Hamming.csv",index=False)

re5=[]
for i in range(10):
    x1,y1=K[i]
    row=[]
    p=np.array([x1,y1])
    for j in range(10):
        x2,y2=K[j]
        q=np.array([x2,y2])
        dist = pdist([p,q], 'seuclidean', V=None)
        row.append(round(dist,3))
    re5.append(row)

dfs = pd.DataFrame(re5)
dfs.to_csv("resultstdeuclid.csv",index=False)








