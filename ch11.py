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

distorsions = []
for i in range(1, 11):
  km = KMeans(n_clusters=i,
              init="k-means++",
              n_init=10,
              max_iter=300,
              random_state=0)
  km.fit(X)
  distorsions.append(km.inertia_)
plt.plot(range(1, 11), distorsions, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")

plt.show()


print(50 * '=')
print('Section: Quantifying the quality of clustering via silhouette plots')
print(50 * '-')

km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric="euclidean")
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
  c_silhouette_vals = silhouette_vals[y_km == c]
  c_silhouette_vals.sort()
  y_ax_upper += len(c_silhouette_vals)
  color = cm.jet(i / n_clusters)
  plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
          edgecolor="none", color=color)
  yticks.append((y_ax_lower + y_ax_upper) / 2.)
  y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel("Cluster")
plt.xlabel("Silhouette coefficient")

plt.show()
print("A bad clunstering: ")

km = KMeans(n_clusters=2,
            init="k-means++",
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

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            s=250, marker="*", c="red", label="centroids")
plt.legend()
plt.grid()
plt.show()

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric="euclidean")
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
  c_silhouette_vals = silhouette_vals[y_km == c]
  c_silhouette_vals.sort()
  y_ax_upper += len(c_silhouette_vals)
  color = cm.jet(i / n_clusters)
  plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
            edgecolor="none", color=color)

  yticks.append((y_ax_lower + y_ax_upper) / 2.)
  y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel("Cluster")
plt.xlabel("Silhouette coefficient")

plt.show()


print(50 * '=')
print('Section: Organizing clusters as a hierarchical tree')
print(50 * '-')

np.random.seed(123)

variables = ["X", "Y", "Z"]
labels = ["ID_0", "ID_1", "ID_2", "ID_3", "ID_4"]

X = np.random.random_sample([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)
print("DataFrame:\n\n", df)

print(50 * '=')
print('Section: Performing hierarchical clustering on a distance matrix')
print(50 * '-')

row_dist = pd.DataFrame(squareform(pdist(df, metric="euclidean")),
                        columns=labels,
                        index=labels)
print("Row distances:\n\n", row_dist)

print("1. incorrect approach: Squareform distance matrix")

row_clusters = linkage(row_dist, method="complete", metric="euclidean")
df1 = pd.DataFrame(row_clusters,
                    columns=["row label 1", "row label 2",
                              "distance", "no. of items in clust."],
                    index=["cluster %d" % (i + 1)
                          for i in range(row_clusters.shape[0])])

print("2. correct approach: Condensed distance matrix")
row_clusters = linkage(pdist(df, metric="euclidean"), method="complete")
df2 = pd.DataFrame(row_clusters,
                  columns=["row label 1", "row label 2",
                            "distance", "no. of items in clust."],
                  index=["cluster %d" % (i + 1)
                        for i in range(row_clusters.shape[0])])

print("3. correct approach: Input sample matrix")

row_clusters = linkage(df.values, method="complete", metric="euclidean")
df3 = pd.DataFrame(row_clusters,
                    columns=["row label 1", "row label 2",
                              "distance", "no. of items in clust."],
                    index=["cluster %d" % (i + 1)
                          for i in range(row_clusters.shape[0])])

row_dendr = dendrogram(row_clusters,
                        labels=labels,)

plt.ylabel("Euclidean distance")

plt.show()



print(50 * '=')
print('Section: Attaching dendrograms to a heat map')
print(50 * '-')

fig = plt.figure(figsize=(8, 8), facecolor="white")
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])

row_dendr = dendrogram(row_clusters, orientation="left")

df_rowclust = df.ix[row_dendr["leaves"][::-1]]

axd.set_xticks([])
axd.set_yticks([])

for i in axd.spines.values():
  i.set_visible(False)

axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation="nearest", cmap="hot_r")
fig.colorbar(cax)
axm.set_xticklabels([""] + list(df_rowclust.columns))
axm.set_yticklabels([""] + list(df_rowclust.index))

plt.show()



print(50 * '=')
print('Section: Applying agglomerative clustering via scikit-learn')
print(50 * '-')

ac = AgglomerativeClustering(n_clusters=2,
                            affinity="euclidean",
                            linkage="complete")

labels = ac.fit_predict(X)
print("Cluster labels: %s" % labels)

print(50 * '=')
print('Section: Attaching dendrograms to a heat map')
print(50 * '-')

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.scatter(X[:, 0], X[:, 1])

plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)
ax1. scatter(X[y_km == 0, 0], X[y_km == 0, 1],
              c="lightblue", marker="o", s=40, label="cluster 1")
ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
            c="red", marker="s", s=40, label="cluster 2")
ax1.set_title("K-means clustering")

ac = AgglomerativeClustering(n_clusters=2,
                              affinity="euclidean",
                              linkage="complete")
y_ac = ac.fit_predict(X)
ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c="lightblue",
            marker="o", s=40, label="cluster 1")
ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c="red",
            marker="s", s=40, label="cluster 2")
ax2.set_title("Agglomerative custering")

plt.legend()

plt.show()
print("DBSCAN")

db = DBSCAN(eps=0.2, min_samples=5, metric="euclidean")
y_db = db.fit_predict(X)
plt.scatter(X[y_db == 0, 0], X[y_db == 0, 1],
            c="lightblue", marker="o", s=40,
            label="cluster 1")
plt.scatter(X[y_db == 1, 0], X[y_db == 1, 1],
            c="red", marker="s", s=40,
            label="cluster 2")

plt.legend()

plt.show()






























