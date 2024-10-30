import numpy as np
import pandas as pd


def gini_criterion(node_data, class_col_name):
    result = 0
    for i in node_data[class_col_name].unique():
        p = (node_data[class_col_name] == i).sum() / node_data.shape[0]
        result += p * (1 - p)
    return result


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

