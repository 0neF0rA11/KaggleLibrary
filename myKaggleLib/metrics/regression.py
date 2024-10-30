import numpy as np


def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)
