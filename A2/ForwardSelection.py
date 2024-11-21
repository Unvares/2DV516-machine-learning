import numpy as np
from ROCAnalysis import ROCAnalysis


class ForwardSelection:
    """
    A class for performing forward feature selection based on maximizing the F-score of a given model.

    Attributes:
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        model (object): Machine learning model with `fit` and `predict` methods.
        selected_features (list): List of selected feature indices.
        best_cost (float): Best F-score achieved during feature selection.
    """

    def __init__(self, X, y, model, seed):
        """
        Initializes the ForwardSelection object.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            model (object): Machine learning model with `fit` and `predict` methods.
            seed (int): Random seed for reproducibility.
        """
        self.X = X
        self.y = y
        self.model = model
        self.seed = seed
        self.selected_features = []
        self.best_cost = -np.inf

    def create_split(self, X, y):
        """
        Creates a train-test split of the data.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.

        Returns:
            X_train (array-like): Features for training.
            X_test (array-like): Features for testing.
            y_train (array-like): Target labels for training.
            y_test (array-like): Target labels for testing.
        """
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        split_index = int(num_samples * 0.8)
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test

    def train_model_with_features(self, features):
        """
        Trains the model using selected features and evaluates it using ROCAnalysis.

        Parameters:
            features (list): List of feature indices.

        Returns:
            float: F-score obtained by evaluating the model.
        """
        X_train, X_test, y_train, y_test = self.create_split(
            self.X[:, features], self.y
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        y_pred = (y_pred >= 0.5).astype(int)
        y_test = (y_test >= 0.5).astype(int)

        roc = ROCAnalysis(y_pred, y_test)
        return roc.f_score()

    def forward_selection(self):
        """
        Performs forward feature selection based on maximizing the F-score.
        """
        current_features = []
        all_combinations = []
        remaining_features = list(range(self.X.shape[1]))

        while remaining_features:
            scores = []
            for feature in remaining_features:
                iteration_features = current_features + [feature]
                score = self.train_model_with_features(iteration_features)
                scores.append((score, feature))
            best_score, best_feature = max(scores, key=lambda x: x[0])
            print(
                f"Current features: {current_features}. Best feature to add: {best_feature} with F-score: {best_score:.3f}"
            )
            current_features.append(best_feature)
            remaining_features.remove(best_feature)
            all_combinations.append((current_features.copy(), best_score))
        best_combination = max(all_combinations, key=lambda x: x[1])
        self.selected_features, self.best_cost = best_combination
        self.fit(self.selected_features)

    def fit(self, selected_features):
        """
        Fits the model using the selected features.
        """
        self.model.fit(self.X[:, selected_features], self.y)
        return self

    def predict(self, X_test):
        """
        Predicts the target labels for the given test features.

        Parameters:
            X_test (array-like): Test features.

        Returns:
            array-like: Predicted target labels.
        """
        # Ensure the model has been trained
        if not hasattr(self, "model"):
            raise ValueError(
                "The model has not been trained yet. Please call the 'fit' method first."
            )
        y_pred = self.model.predict(X_test)

        return y_pred

