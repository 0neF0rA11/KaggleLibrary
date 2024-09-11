import numpy as np


class MSE:
    def calculate_error(self, X, y, weights):
        return np.mean((np.dot(X, weights) - y) ** 2)

    def derivative_function(self, X, y, weights):
        return np.dot(X.T, (np.dot(X, weights) - y)) / len(y)