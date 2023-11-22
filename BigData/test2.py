import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
import random


def most_common(lst):
    """
    Return the most frequently occuring element in a list.
    """
    return max(set(lst), key=lst.count)


# def euclidean(point, data):
#     """
#     Return euclidean distances between a point & a dataset
#     """
#     return np.sqrt(np.sum((point - data)**2, axis=1))
def euclidean(point, data):
    """
    Return euclidean distances between a point & a dataset
    """
    # Flatten the dataset into a list of values (ignoring the tuple/list structure)
    flat_data = [val for sublist in data for val in sublist]
    
    # Flatten the point into a list of values (ignoring the tuple/list structure)
    flat_point = [val for val in point]
    
    # Calculate the squared difference between corresponding values in the flattened point and data
    # squared_diff = [(flat_point[i] - flat_data[i]) ** 2 for i in range(len(flat_data))]
    squared_diff = [np.square(np.subtract(flat_point[0], flat_data[0])), np.square(np.subtract(flat_point[1], flat_data[1]))]

    # Return the square root of the sum of squared differences
    return np.sqrt(np.sum(squared_diff))

class KMeans:

    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):

        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        self.centroids = [random.choice(X_train)]

        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
            # Normalize the distances
            dists /= np.sum(dists)
            # Choose remaining points based on their distances
            new_centroid_idx = np.random.choice(range(len(X_train)), size=1, p=dists)[0]  # Indexed @ zero to get val, not array of val
            self.centroids += [X_train[new_centroid_idx]]

        # This method of randomly selecting centroid starts is less effective
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]

        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)

        return centroids, centroid_idxs


# Create a dataset of 2D distributions

with open('Clustering_dataset.txt') as file:
    lines = file.readlines()
    X_train = [list(map(int, element.split())) for element in lines]


# # ========================================================================= test begin
# centroids = [random.choice(X_train)]
# flat_data = [val for sublist in X_train for val in sublist]
# flat_point = [val for val in centroids]
# print(f"point: {flat_point[i]} and data: {flat_data[i]}" for i in range(len(flat_data)))
# # ========================================================================= test over


# Fit centroids to dataset
kmeans = KMeans()
kmeans.fit(X_train)

# View results
class_centers, classification = kmeans.evaluate(X_train)

plt.plot([x for x, _ in kmeans.centroids],
         [y for _, y in kmeans.centroids],
         '+',
         markersize=10,
         )
plt.title("k-means")
plt.show()
