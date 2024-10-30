import numpy as np
from myKaggleLib.losses.regression import MSELoss
from myKaggleLib.optimizers import GD, SGD


class LinearRegression:
    def __init__(self, loss='MSE', optim='GD'):
        self.weights = None
        self.optim = optim

        if loss == 'MSE':
            self.loss = MSELoss()
        if optim == 'GD':
            self.optim = GD()
        elif optim == 'SGD':
            self.optim = SGD()

    def __initialize_weights(self, n_features):
        self.weights = np.zeros(n_features + 1)

    def fit(self, X, y, lr=0.01, epochs=1000):
        n_samples, n_features = X.shape
        X = np.c_[np.ones(X.shape[0]), X]

        self.__initialize_weights(n_features)

        self.optim.initialize(self.weights,
                              lr=lr,
                              derivative_function=self.loss.derivative_function)

        for epoch in range(1, epochs + 1):
            self.weights = self.optim.step(X, y)
            if epoch % 100 == 0:
                loss = self.loss.calculate_error(X, y, self.weights)
                print(f'Epoch {epoch}: Loss = {loss}')

    def predict(self, X):
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X_with_bias, self.weights)

    def get_weights_and_bias(self):
        return self.weights[1:], self.weights[0]
