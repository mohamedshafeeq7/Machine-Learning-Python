import numpy as np

class KMeans:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for _ in range(self.max_iterations):
            # Assign each data point to the nearest centroid
            labels = self._assign_labels(X)

            # Update centroids based on the mean of the data points assigned to each centroid
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            # Check if centroids have converged
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return labels

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

# Example usage:
X = np.array([[1, 2], [1, 3], [2, 2], [7, 8], [8, 8], [8, 7]])
kmeans = KMeans(k=2)
labels = kmeans.fit(X)

print("Cluster labels:", labels)
