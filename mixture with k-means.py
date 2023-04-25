import numpy as np
import matplotlib.pyplot as plt
n_samples = 10000 # The number of data points to generate
# Set of parameters needed for the 2 Gaussians
mu1 = -2
mu2 = 3
sigma1 = 1
sigma2 = 2
weight1 = 0.5
weight2 = 0.8
data = np.zeros(n_samples) # Generate data from the mixture model
for i in range(n_samples):
    if np.random.uniform() < weight1:
        data[i] = np.random.normal(mu1, sigma1)
    else:
        data[i] = np.random.normal(mu2, sigma2)
plt.hist(data, bins=30, density=True) #plot the data
plt.show()
n_clusters = 2 # Initialize the K-means algorithm
centroids = np.random.uniform(low=min(data), high=max(data), size=n_clusters)
old_centroids = np.zeros(n_clusters)
clusters = np.zeros(n_samples)
while not np.allclose(centroids, old_centroids): # Run the K-means algorithm
    old_centroids = centroids.copy()
    for i in range(n_samples): # Assign each data point to the closest centroid
        distances = np.abs(data[i] - centroids)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    for i in range(n_clusters): # Update the centroids to the mean of the assigned data points
        centroids[i] = np.mean(data[clusters == i])
plt.hist(data[clusters == 0], bins=30, density=True, alpha=0.5, label='Cluster 1') # Plot the K-means algorithm
plt.hist(data[clusters == 1], bins=30, density=True, alpha=0.5, label='Cluster 2')
plt.scatter(centroids, [0, 0], marker='x', color='red', linewidths=2, label='Centroids')
plt.legend()
plt.show()
