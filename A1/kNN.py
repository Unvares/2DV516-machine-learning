import numpy as np


def kNN_predict(T, X, k):
    # extracting two input values and the label
    x1 = T[:, 0]
    x2 = T[:, 1]
    z = T[:, 2]

    # calculating distances for every given point
    distances = np.empty((len(X), len(T)), float)
    for i, point in enumerate(X):
        # calculating Euclidian Distance
        distances[i] = np.sqrt((x1 - point[0]) ** 2 + (x2 - point[1]) ** 2)

    labels = np.empty(len(X), float)
    for i, point_distances in enumerate(distances):
        k_closest_indices = np.argpartition(point_distances, k)[:k]
        label = round(z[k_closest_indices].mean())
        labels[i] = label
    return labels
