from abc import ABC, abstractmethod
import numpy as np


class MachineLearningModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        pass

    def normalize(self, X):
        """
        Normalize the input features using standard deviation.
        Works both for matrices and vectors.

        Parameters:
        X (array-like): Features of the data

        Returns:
        array-like: The normalized data.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        # Ensure self.mean and self.std are always arrays
        self.mean = np.atleast_1d(self.mean)
        self.std = np.atleast_1d(self.std)

        # Set intercepts equal to 1 (otherwise they will not be counted at all)
        self.std[self.std == 0] = 1
        X_normalized = np.where(self.std != 1, (X - self.mean) / self.std, X)
        return X_normalized

    def _polynomial_features(self, X):
        """
        Generate polynomial features from the input features.
        Check the slides for hints on how to implement this one.
        This method is used by the regression models and must work
        for any degree polynomial
        Parameters:

        X (array-like): Features of the data.

        Returns:
        X_poly (array-like): Polynomial features.
        """
        # Ensure X is 2D if it's not already
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.hstack(
            [np.ones((X.shape[0], 1))] + [X**i for i in range(1, self.degree + 1)]
        )

    def score(self, X, y):
        y_pred = self.predict(X)
        tss = np.sum((y - np.mean(y)) ** 2)
        rss = np.sum((y - y_pred) ** 2)
        r2 = 1 - (rss / tss)
        return r2


class RegressionModelNormalEquation(MachineLearningModel):
    """
    Class for regression models using the Normal Equation for polynomial regression.
    """

    def __init__(self, degree):
        """
        Initialize the model with the specified polynomial degree.

        Parameters:
        degree (int): Degree of the polynomial features.
        """
        self.degree = degree

    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        X_poly = self._polynomial_features(X)
        self.theta = np.zeros(X_poly.shape[1])
        XtX_inverse = np.linalg.inv(np.dot(X_poly.T, X_poly))
        Xt_y = np.dot(X_poly.T, y)
        self.theta = np.dot(XtX_inverse, Xt_y)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        X_poly = self._polynomial_features(X)
        return np.dot(X_poly, self.theta)

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        y_predicted = self.predict(X)
        return np.mean((y_predicted - y) ** 2)


class RegressionModelGradientDescent(MachineLearningModel):
    """
    Class for regression models using gradient descent optimization.
    """

    def __init__(self, degree, learning_rate=0.01, num_iterations=3000):
        """
        Initialize the model with the specified parameters.

        Parameters:
        degree (int): Degree of the polynomial features.
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for gradient descent.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        self.cost_history = []
        X_poly = self._polynomial_features(X)
        self.theta = np.zeros(X_poly.shape[1])

        for _ in range(self.num_iterations):
            residuals = np.dot(X_poly, self.theta) - y
            gradient = np.dot(X_poly.T, residuals)
            self.theta = self.theta - self.learning_rate * gradient

            MSE = np.mean(residuals**2)
            self.cost_history.append(MSE)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        X_poly = self._polynomial_features(X)
        return np.dot(X_poly, self.theta)

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        y_predicted = self.predict(X)
        return np.mean((y_predicted - y) ** 2)


class LogisticRegression(MachineLearningModel):
    """
    Logistic Regression model using gradient descent optimization.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the logistic regression model.

        Parameters:
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        self.degree = 1
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.convergence_iteration = 0

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        self.cost_history = []
        X_poly = self._polynomial_features(X)
        self.theta = np.zeros(X_poly.shape[1])
        tolerance = 1e-8  # Convergence threshold
        check_interval = 3  # Number of last values to check

        for i in range(self.num_iterations):
            z = X_poly @ self.theta
            h = self._sigmoid(z)
            gradient = (X_poly.T @ (h - y)) / len(y)
            self.theta -= self.learning_rate * gradient

            cost = self._cost_function(X_poly, y)
            self.cost_history.append(cost)

            if i >= check_interval and self.convergence_iteration == 0:
                recent_costs = self.cost_history[-check_interval:]
                if np.std(recent_costs) < tolerance:
                    self.convergence_iteration = i

    def predict(self, X):
        """
        Make predictions using the trained logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        X_poly = self._polynomial_features(X)
        z = X_poly @ self.theta
        return self._sigmoid(z)

    def evaluate(self, X, y):
        """
        Evaluate the logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (e.g., accuracy).
        """
        predictions = self.predict(X)
        predictions = (predictions >= 0.5).astype(int)
        return np.mean(predictions == y)

    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        z = np.clip(z, -500, 500)  # Clip values to prevent overflow
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        epsilon = 1e-8  # Small value to avoid log(0)
        z = X @ self.theta
        h = self._sigmoid(z)
        h = np.clip(h, epsilon, 1 - epsilon)
        return -np.mean(y.T * np.log(h) + (1 - y).T * np.log(1 - h))


class NonLinearLogisticRegression:
    """
    Nonlinear Logistic Regression model using gradient descent optimization.
    It works for 2 features (when creating the variable interactions)
    """

    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the nonlinear logistic regression model.

        Parameters:
        degree (int): Degree of polynomial features.
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.convergence_iteration = 0

    def fit(self, X, y):
        """
        Train the nonlinear logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        self.cost_history = []
        X_poly = self.mapFeature(X[:, 0], X[:, 1], self.degree)
        self.theta = np.zeros(X_poly.shape[1])
        tolerance = 1e-8  # Convergence threshold
        check_interval = 3  # Number of last values to check

        for i in range(self.num_iterations):
            z = X_poly @ self.theta
            h = self._sigmoid(z)
            gradient = (X_poly.T @ (h - y)) / len(y)
            self.theta -= self.learning_rate * gradient

            cost = self._cost_function(X_poly, y)
            self.cost_history.append(cost)

            if i >= check_interval and self.convergence_iteration == 0:
                recent_costs = self.cost_history[-check_interval:]
                if np.std(recent_costs) < tolerance:
                    self.convergence_iteration = i

    def predict(self, X):
        """
        Make predictions using the trained nonlinear logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        X_poly = self.mapFeature(X[:, 0], X[:, 1], self.degree)
        z = X_poly @ self.theta
        return self._sigmoid(z)

    def evaluate(self, X, y):
        """
        Evaluate the nonlinear logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        predictions = self.predict(X)
        predictions = (predictions >= 0.5).astype(int)
        return np.mean(predictions == y)

    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        z = np.clip(z, -500, 500)  # Clip values to prevent overflow
        return 1 / (1 + np.exp(-z))

    def mapFeature(self, X1, X2, D):
        """
        Map the features to a higher-dimensional space using polynomial features.
        Check the slides to have hints on how to implement this function.
        Parameters:
        X1 (array-like): Feature 1.
        X2 (array-like): Feature 2.
        D (int): Degree of polynomial features.

        Returns:
        X_poly (array-like): Polynomial features.
        """
        one = np.ones([len(X1), 1])
        X_poly = np.c_[one, X1, X2]
        for i in range(2, D + 1):
            for j in range(0, i + 1):
                Xnew = X1 ** (i - j) * X2**j
                Xnew = Xnew.reshape(-1, 1)
                X_poly = np.append(X_poly, Xnew, 1)
        return X_poly

    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        epsilon = 1e-8  # Small value to avoid log(0)
        z = X @ self.theta
        h = self._sigmoid(z)
        h = np.clip(h, epsilon, 1 - epsilon)
        return -np.mean(y.T * np.log(h) + (1 - y).T * np.log(1 - h))
