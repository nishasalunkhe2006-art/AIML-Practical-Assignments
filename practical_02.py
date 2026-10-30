import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample dataset
X, y = make_blobs(
    n_samples=100,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# Get cluster labels and centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Display results
print("K-Means Clustering Results")
print("Total Data Points:", len(X))
print("Number of Clusters:", kmeans.n_clusters)
print("\nCluster Labels:")
print(labels)

print("\nCluster Centers:")
print(centers)

# Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Cluster Centers"
)

plt.title("K-Means Clustering using Unsupervised Learning")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()