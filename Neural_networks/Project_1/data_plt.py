###############
##### Classificação binária com modelo de rede neural - backprop / regressão logística
###############

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("Neural_networks/classification2.txt", header=None)

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
pos , neg = (y==1).reshape(118,1) , (y==0).reshape(118,1)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(["Accepted","Rejected"],loc=0)
plt.show()