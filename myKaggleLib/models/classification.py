import numpy as np
from myKaggleLib.losses.classification import BCELoss
from myKaggleLib.optimizers import SGD, GD


class LogisticRegression:
    def __init__(self, loss='binary_cross_entropy_l2', optim='SGD', alpha=0.01):
        self.weights = None

        if loss == 'binary_cross_entropy':
            self.loss = BCELoss()
        elif loss == 'binary_cross_entropy_l1':
            self.loss = BCELoss('L1', alpha=alpha)
        elif loss == 'binary_cross_entropy_l2':
            self.loss = BCELoss('L2', alpha=alpha)

        if optim == 'GD':
            self.optim = GD()
        elif optim == 'SGD':
            self.optim = SGD()

    def __sigmoid(self, X, weights):
        return 1 / (1 + np.exp(-np.dot(X, weights)))

    def __initialize_weights(self, n_features):
        self.weights = np.zeros(n_features + 1)

    def fit(self, X, y, lr=0.01, epochs=1000):
        n_samples, n_features = X.shape
        X = np.c_[np.ones(X.shape[0]), X]

        self.__initialize_weights(n_features)

        self.optim.initialize(self.weights,
                              lr=lr,
                              derivative_function=self.loss.derivative_function
                              )

        for epoch in range(1, epochs + 1):
            self.weights = self.optim.step(X, y)
            if epoch % 100 == 0:
                loss = self.loss.calculate_error(X, y, self.weights)
                print(f'Epoch {epoch}: Loss = {loss}')

    def predict(self, X, threshold=0.5):
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        prob = self.__sigmoid(X_with_bias, self.weights)
        return np.where(prob >= threshold, 1, 0)

    def get_weights_and_bias(self):
        return self.weights[1:], self.weights[0]