import numpy as np


class BCELoss:
    def __init__(self, type_reg=None, alpha=0.01):
        self.type_reg = type_reg
        self.alpha = alpha

    def __sigmoid(self, X, weights):
        return 1 / (1 + np.exp(-np.dot(X, weights)))

    def calculate_error(self, X, y, weights):
        p = self.__sigmoid(X, weights)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        if self.type_reg == 'L1':
            loss += self.alpha * np.sum(np.abs(weights))
        elif self.type_reg == 'L2':
            loss += self.alpha * np.sum(weights ** 2)

        return loss

    def derivative_function(self, X, y, weights):
        gradient = np.dot(X.T, (self.__sigmoid(X, weights) - y)) / X.shape[0]

        if self.type_reg == 'L1':
            gradient += self.alpha * np.sign(weights)
        elif self.type_reg == 'L2':
            gradient += 2 * self.alpha * weights

        return gradient
