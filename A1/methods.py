import numpy as np


def kNN_predict(T, X, k):
    labels = T[:, 2]
    num_points = len(X)
    num_labels = len(T)

    point_distances = np.empty((num_points, num_labels))
    for i, point in enumerate(X):
        point_distances[i] = calc_distance_eucledian(T, point)

    predicted_labels = np.empty(num_points, dtype=float)

    # iterate over each point's distances
    for i, distances in enumerate(point_distances):
        k_closest_indices = np.argpartition(distances, k)[:k]
        predicted_label = round(np.mean(labels[k_closest_indices]))
        predicted_labels[i] = predicted_label

    return predicted_labels


def kNN_regression(X, line, k):
    x_values = X[:, 0]
    y_values = X[:, 1]

    y_line = np.empty(len(line), float)
    for i, x_line in enumerate(line):
        distances = calc_distance_D1(x_values, x_line)
        k_closest_indices = np.argpartition(distances, k)[:k]
        predicted_y_value = y_values[k_closest_indices].mean()
        y_line[i] = predicted_y_value
    return y_line


def normalize_features(X):
    normalized_features = []
    for i in range(X.shape[1]):
        xi = X[:, i]
        xi_mean = xi.mean()
        xi_standard_deviation = np.sqrt(np.square(xi - xi_mean).mean())
        xi_normalized = (xi - xi_mean) / xi_standard_deviation
        normalized_features.append(xi_normalized)
    return np.column_stack(normalized_features)


def calc_distance_D1(X, point):
    return np.abs(X - point)


# TODO: calculation for all the points breaks kNN_predict function
def calc_distance_eucledian(X, point):
    x1, x2 = X[:, 0], X[:, 1]
    point1, point2 = point[0], point[1]
    return np.sqrt((x1 - point1) ** 2 + (x2 - point2) ** 2)


def calc_MSE(X, X_hat):
    total_error = 0
    for point in X:
        x_distances = calc_distance_D1(X_hat[:, 0], point[0])
        closest_point_index = np.argmin(x_distances)
        closest_point = X_hat[closest_point_index]
        total_error += np.square(point[1] - closest_point[1])
    return total_error / len(X)
