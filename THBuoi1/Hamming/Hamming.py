import pandas as pd 
import numpy as np 
import random
from scipy.spatial import distance


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
        dist = distance.hamming(p,q)
        row.append(round(dist,3))
    re.append(row)

dfs = pd.DataFrame(re)
dfs.to_csv("result Hamming.csv",index=False)
re=[]
for i in range(10):
    x1,y1=K[i]
    row=[]
    p=np.array([x1,y1])
    for j in range(10):
        x2,y2=K[j]
        q=np.array([x2,y2])
        dist = distance.hamming(p,q)
        row.append(round(dist,3))
    re.append(row)

dfs = pd.DataFrame(re)
dfs.to_csv("result Hamming10.csv",index=False)

re=[]
for i in range(100):
    x1,y1=K[i]
    row=[]
    p=np.array([x1,y1])
    for j in range(100):
        x2,y2=K[j]
        q=np.array([x2,y2])
        dist = distance.hamming(p,q)
        row.append(round(dist,3))
    re.append(row)

dfs = pd.DataFrame(re)
dfs.to_csv("result Hamming100.csv",index=False)

re=[]
for i in range(1000):
    x1,y1=K[i]
    row=[]
    p=np.array([x1,y1])
    for j in range(1000):
        x2,y2=K[j]
        q=np.array([x2,y2])
        dist = distance.hamming(p,q)
        row.append(round(dist,3))
    re.append(row)

dfs = pd.DataFrame(re)
dfs.to_csv("result Hamming1000.csv",index=False)

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
        dist = distance.hamming(p,q)
        row.append(round(dist,3))
    re.append(row)

dfs = pd.DataFrame(re)
dfs.to_csv("result Hamming200.csv",index=False)

