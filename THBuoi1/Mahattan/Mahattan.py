import pandas as pd 
import numpy as np 
from scipy.spatial import distance
import random

dataset =pd.read_csv("Point_Data.csv")
K=[]
for i in dataset.itertuples():
    K.append((round(i.x,3),round(i.y,3)))
re=[]
for i in range(10):
    x1,y1=K[i]
    row=[]
    p=np.array([x1,y1])
    for j in range(10):
        x2,y2=K[j]
        q=np.array([x2,y2])
        dist = distance.cityblock(p,q)
        row.append(round(dist,3))
    re.append(row)

dfs = pd.DataFrame(re)
dfs.to_csv("resultMahattan10.csv",index=False)

re=[]
for i in range(100):
    x1,y1=K[i]
    row=[]
    p=np.array([x1,y1])
    for j in range(100):
        x2,y2=K[j]
        q=np.array([x2,y2])
        dist = distance.cityblock(p,q)
        row.append(round(dist,3))
    re.append(row)

dfs = pd.DataFrame(re)
dfs.to_csv("resultMahattan100.csv",index=False)

re=[]
for i in range(1000):
    x1,y1=K[i]
    row=[]
    p=np.array([x1,y1])
    for j in range(1000):
        x2,y2=K[j]
        q=np.array([x2,y2])
        dist = distance.cityblock(p,q)
        row.append(round(dist,3))
    re.append(row)

dfs = pd.DataFrame(re)
dfs.to_csv("resultMahattan1000.csv",index=False)

re=[]
list = [i for i in range(len(K))]
list_rd = random.sample(list, 200)
for i in range(200):
    x1,y1=K[list_rd[i]]
    
    row=[]
    p=np.array([x1,y1])
    for j in range(200):
        x2,y2=K[list_rd[j]]
        q=np.array([x2,y2])
        dist = distance.cityblock(p,q)
        row.append(round(dist,3))
    re.append(row)

dfs = pd.DataFrame(re)
dfs.to_csv("resultMahattan200.csv",index=False)
