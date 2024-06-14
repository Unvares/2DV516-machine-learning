import numpy as np


class kNNModel:
    def normalize(self, test=True, train=True, type=None):
        if not test and not train:
            raise ValueError("Both 'test' and 'train' cannot be False.")
        if type is None:
            raise ValueError("'type' cannot be None.")

        if type == "sd":
            self.standard_deviation(test=test, train=train)

    def standard_deviation(self, test=False, train=False):
        if test and train:
            for D in (self.X_train, self.X_test):
                normalized_features = np.empty(D.shape)
                for i in range(D.shape[1]):
                    xi = D[:, i]
                    xi_mean = xi.mean()
                    xi_std_dev = np.sqrt(np.square(xi - xi_mean).mean())
                    xi_normalized = (xi - xi_mean) / xi_std_dev
                    normalized_features[:, i] = xi_normalized
                if D is self.X_train:
                    self.X_train = normalized_features
                elif D is self.X_test:
                    self.X_test = normalized_features

    def fit(self, X, y, X_test=None, y_test=None):
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "Cannot instantiate: Number of data points ",
                "is different from the number of labels.",
            )
        self.X_train = X
        self.y_train = y
        if X_test is None:
            self.X_test = X
        else:
            self.X_test = X_test

        if y_test is None:
            self.y_test = y
        else:
            self.y_test = y_test

    def classify(self, X, k=5):
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                "Number of input features is different from",
                "the number of features in the training set",
            )

        num_points = len(X)
        num_labels = len(self.X_train)
        point_distances = np.empty((num_points, num_labels))
        for i, point in enumerate(X):
            point_distances[i] = self.euclidian_distance(self.X_train, point)

        predicted_labels = np.empty(num_points, dtype=float)
        for i, distances in enumerate(point_distances):
            k_closest_indices = np.argpartition(distances, k)[:k]
            labels, counts = np.unique(
                self.y_train[k_closest_indices], return_counts=True
            )
            predicted_labels[i] = labels[np.argmax(counts)]

        return predicted_labels

    def regress(self, X, k=5):
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                "Number of input features is different from",
                "the number of features in the training set",
            )

        predicted_labels = np.empty(len(X), float)
        for i, point in enumerate(X):
            distances = self.euclidian_distance(self.X_train, point)
            k_closest_indices = np.argpartition(distances, k)[:k]
            predicted_labels[i] = self.y_train[k_closest_indices].mean()
        return predicted_labels

    def euclidian_distance(self, X, point):
        diff = X - point
        num_features = X.shape[1]
        if num_features == 1 or num_features == 2:
            return np.sqrt(np.sum(diff**2, axis=1))
        else:
            return np.power(np.sum(diff**2, axis=1), 1 / num_features)

    def MSE(self, X, y, test=True):
        if test:
            D_features = self.X_test
            D_labels = self.y_test
        else:
            D_features = self.X_train
            D_labels = self.y_train

        total_error = 0
        for i, point in enumerate(D_features):
            distances = self.euclidian_distance(X, point)
            closest_point_index = np.argmin(distances)
            closest_point_label = y[closest_point_index]
            total_error += np.square(D_labels[i] - closest_point_label)
        return total_error / len(D_features)
