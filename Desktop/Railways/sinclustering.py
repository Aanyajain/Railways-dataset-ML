import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv(r"C:\Users\om\Desktop\sinproject\d.csv")
from sklearn.preprocessing import LabelEncoder
l_x=LabelEncoder()
data['Train Name']=l_x.fit_transform(data['Train Name'])
l_x1=LabelEncoder()
data['Station Code']=l_x1.fit_transform(data['Station Code'])
l_x2=LabelEncoder()
data['Station Name']=l_x2.fit_transform(data['Station Name'])
l_x3=LabelEncoder()
data['Source Station']=l_x3.fit_transform(data['Source Station'])
l_x4=LabelEncoder()
data['Source Station Name']=l_x4.fit_transform(data['Source Station Name'])
l_x5=LabelEncoder()
data['Destination Station']=l_x5.fit_transform(data['Destination Station'])
l_x6=LabelEncoder()
data['Destination Station Name']=l_x6.fit_transform(data['Destination Station Name'])

x=data.iloc[:,[6,9]].values
#kmeans clustering

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,9):
    km=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)
plt.plot(range(1,9),wcss)
plt.title("elbow method")
plt.xlabel("no of clusters")
plt.ylabel("wcss")
plt.show()
#k=4
km=KMeans(n_clusters=4,init='k-means++',max_iter=300,random_state=0)
y_km=km.fit_predict(x)
plt.scatter(x[y_km==0,0],x[y_km==0,1],s=100,c="red",label="cluster1")
plt.scatter(x[y_km==1,0],x[y_km==1,1],s=100,c="blue",label="cluster2")
plt.scatter(x[y_km==2,0],x[y_km==2,1],s=100,c="green",label="cluster3")
plt.scatter(x[y_km==3,0],x[y_km==3,1],s=100,c="cyan",label="cluster4")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=300,c='yellow',label='centroids')
plt.title("cluster of values")
plt.xlabel("source station")
plt.ylabel("distance")
plt.legend()
plt.show()

#hierarchical clustering
x=data.iloc[:,[6,9]].values
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title("dendrogram")
plt.xlabel("source station")
plt.ylabel("euclidean distances")
plt.show()
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)

plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c="red",label="cluster1")
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c="blue",label="cluster2")
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c="green",label="cluster3")
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c="cyan",label="cluster4")
plt.title("cluster of values")
plt.xlabel("source station")
plt.ylabel("distance")
plt.legend()
plt.show()
