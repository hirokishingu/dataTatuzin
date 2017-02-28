import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN



print(50 * '=')
print('Section: Grouping objects by similarity using k-means')
print(50 * '-')

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

plt.scatter(X[:, 0], X[:, 1], c="cyan", marker="o", s=50)
plt.grid()

plt.show()

km = KMeans(n_clusters=3,
            init="random",
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)

y_km = km.fit_predict(X)

plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50,
            c="lightgreen",
            marker="s",
            label="cluster 1")

plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50,
            c="orange",
            marker="o",
            label="cluster 2")
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50,
            c="lightblue",
            marker="v",
            label="cluster 3")
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker="*",
            c="red",
            label="centroids")
plt.legend()
plt.grid()
plt.show()


print(50 * '=')
print('Section: Using the elbow method to find the optimal number of clusters')
print(50 * '-')

print('Distortion: %.2f' % km.inertia_)










































