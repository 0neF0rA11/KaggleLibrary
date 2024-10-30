import numpy as np
import pandas as pd
from myKaggleLib.losses.classification import BCELoss, gini_criterion
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


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def best_split(self, data, class_col_name):
        best_ig = float('inf')
        best_feature = None
        best_threshold = None
        for feature in data.columns.drop(class_col_name):
            unique_values = sorted(data[feature].unique())
            for threshold in unique_values:
                left_split = data[data[feature] < threshold]
                right_split = data[data[feature] >= threshold]
                if len(left_split) < self.min_samples_split or len(right_split) < self.min_samples_split:
                    continue

                left_gini = gini_criterion(left_split, class_col_name)
                right_gini = gini_criterion(right_split, class_col_name)

                information_gate = (len(left_split) / len(data)) * left_gini + (
                            len(right_split) / len(data)) * right_gini

                if information_gate < best_ig:
                    best_ig = information_gate
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold, best_ig

    def stop(self, depth):
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return False

    def ans(self, data, class_col_name):
        return data[class_col_name].mode()[0]

    def build_tree(self, data, class_col_name, depth=0):
        if len(data[class_col_name].unique()) == 1:
            return self.ans(data, class_col_name)

        feature, threshold, gini = self.best_split(data, class_col_name)

        if feature is None:
            return self.ans(data, class_col_name)

        left_split = data[data[feature] < threshold]
        right_split = data[data[feature] >= threshold]

        if self.stop(depth):
            return self.ans(data, class_col_name)

        left_tree = self.build_tree(left_split, class_col_name, depth + 1)
        right_tree = self.build_tree(right_split, class_col_name, depth + 1)

        return {'feature': feature, 'threshold': threshold, 'left': left_tree, 'right': right_tree}

    def fit(self, X, y):
        X['target_col'] = y
        self.tree = self.build_tree(X, 'target_col')

    def predict_sample(self, sample):
        node = self.tree
        while isinstance(node, dict):
            if sample[node['feature']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node

    def predict(self, data):
        return data.apply(self.predict_sample, axis=1)