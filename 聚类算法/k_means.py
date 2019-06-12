# def k_means(data):
#     m = len(data)
#     n = len(data[0])
#     cluster = [-1 for x in range(m)]   #所有样本尚未聚类
#     cluster_center = [[] for x in range(k)]  # 聚类中心
#     cc = [[] for x in range(k)]  #下一轮的聚类中心
#     c_number = [0 for x in range(k)]  #每个簇中样本的数目

#     #随机选择簇中心
#     i = 0
#     while i<k:
#         j = random.randint(0,m-1)
#         if is_similar(data[i],cluster_center):
#             continue
#         cluster_center[i] = data[j][:]
#         cc[i] = [0 for x in range(n)]
#         i +=1
#     for times in range(40):
#         for i in range(m):
#             c = nearest(data[i],cluster_center)
#             cluster[i] = c   #第i个样本归于第c簇
#             c_number[c] +=1
#             add(cc[c],data[i])
#         for i in range(k):
#             divide(cc[i],c_number[i])
#             c_number[i]=0
#             cluster_center[i]=cc[i][i]
#             zero_list(cc[i])
#         print(times,cluster_center)
#     return cluster
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import sklearn.datasets as ds
from sklearn.neighbors import KNeighborsClassifier
X = np.array([[1,2],[1,4],[1,1],[2,1],[10,2],[10,4],[10,0],[9,4],[8,5]])
model = KMeans(n_clusters=2,random_state=0).fit(X)
model.labels_
y_pred = model.predict(X)
print(y_pred)
plt.scatter(X[:,0],X[:,1],c=y_pred, marker='o')
plt.show()


# Scikit-learn实现聚类算法
x, y = ds.make_blobs(400, n_features=2, centers=4, random_state=2018)
model = KMeans(n_clusters=4, init='k-means++')
model.fit(x)
y_pred = model.predict(x)
plt.subplot(121)
plt.plot(x[:, 0], x[:, 1], 'r.', ms=3)
plt.subplot(122)
plt.scatter(x[:, 0], x[:, 1], c=y_pred, marker='.',cmap=mpl.colors.ListedColormap(list('rgbm')))
plt.tight_layout(2)
plt.show()

# User CF实现
train_x = np.array([[0.238,0,0.1905,0.1905,0.1905,0.1905],
               [0,0.177,0,0.294,0.235,0.294],[0.2,0.16,0.12,0.12,0.2,0.2]])
y = np.array(['B','C','D'])
test_x = [[0.2174,0.2174,0.1304,0.0,0.2174,0.2174]]
model_neigh = KNeighborsClassifier(n_neighbors=3)
model_neigh.fit(train_x,y)
print(model_neigh.predict(test_x))

# Item CF实现
train_x = np.array([[0.417,0.0,0.25,0.333],[0.3,0.4,0.0,0.3],
                    [0.0,0.0,0.625,0.375],[0.278,0.222,0.222,0.278],
                    [0.263,0.211,0.263,0.263]])
test_x = [[0.334,0.333,0.0,0.333]]
y_label = np.array(['B','C','D','E','F'])
model_neigh = KNeighborsClassifier(n_neighbors=3)
model_neigh.fit(train_x,y_label)
print(model_neigh.predict(test_x))
