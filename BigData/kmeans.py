import numpy as np
import random

class KMeans:
    def __init__(self, dataset, K=3):
        self.dataset = dataset
        self.K = K

    def dist(self, x1, x2):
        return np.sqrt(np.sum((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2))

    def get_centroids(self):
        centroids = []
        for i in range(self.K):
            centroids.append(random.choice(self.dataset))
        return centroids

    def get_labels(self):
        labels = []
        centroids = self.get_centroids()
        for data in self.dataset:
            dists = [self.dist(data, centroid) for centroid in centroids]
            labels.append(dists.index(min(dists)))
        return labels

    def cluster(self):
        self.centroids = self.get_centroids()
        while True:
            prev_centroids = self.centroids
            self.labels = self.get_labels()
            self.centroids = []
            for k in range(self.K):
                self.centroids.append(tuple(np.mean([data[i] for data in self.dataset for i in range(len(data)) if self.labels[i] == k], axis=0)))
            if self.has_converged(prev_centroids):
                break
        return self.centroids, self.labels

    def has_converged(self, prev_centroids):
        for i in range(self.K):
            if self.dist(prev_centroids[i], self.centroids[i]) > 1e-5:
                return False
        return True
    
with open('Clustering_dataset.txt', 'r') as file:
    lines = file.readlines()
    tuples = [tuple(map(int, element.split())) for element in lines]
    
def k_means_clustering(dataset, K=3):
    kmeans = KMeans(dataset, K)
    centroids, labels = kmeans.cluster()
    return centroids, labels


with open('Clustering_dataset.txt', 'r') as file:
    lines = file.readlines()
    tuples = [tuple(map(int, element.split())) for element in lines]
    # centroids, labels = k_means_clustering(tuples, K=10)
    # print("Centroids are : ", centroids)
    print(k_means_clustering(tuples))
