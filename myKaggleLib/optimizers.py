import numpy as np


class GD:
    def __init__(self):
        self.derivative_function = None
        self.lr = None
        self.weights = None

    def initialize(self, weights, lr, derivative_function):
        if lr < 0:
            raise ValueError("The learning rate must be a positive number.")
        if not callable(derivative_function):
            raise ValueError("Derivative should be a function that calculates the gradient.")

        self.weights = weights
        self.lr = lr
        self.derivative_function = derivative_function

    def step(self, X, y):
        grad = self.derivative_function(X, y, self.weights)
        self.weights -= self.lr * grad
        return self.weights

    def __repr__(self):
        return f"SGD(weights={self.weights}, lr={self.lr})"


class SGD:
    def __init__(self):
        self.derivative_function = None
        self.lr = None
        self.weights = None

    def initialize(self, weights, lr, derivative_function):
        if lr < 0:
            raise ValueError("The learning rate must be a positive number.")
        if not callable(derivative_function):
            raise ValueError("Derivative should be a function that calculates the gradient.")

        self.weights = weights
        self.lr = lr
        self.derivative_function = derivative_function

    def step(self, X, y, batch_size=16):
        shuffled_indexes = np.random.permutation(X.shape[0])
        shuffled_X, shuffled_y = X[shuffled_indexes], y[shuffled_indexes]
        for i in range(0, X.shape[0], batch_size):
            grad = self.derivative_function(shuffled_X[i:i + batch_size], shuffled_y[i:i + batch_size], self.weights)
            self.weights -= self.lr * grad
        return self.weights

    def __repr__(self):
        return f"SGD(weights={self.weights}, lr={self.lr})"