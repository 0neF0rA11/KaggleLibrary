import numpy as np


def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_true.shape[0]


def precision_score(y_true, y_pred, average='binary'):
    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if tp + fp > 0 else 0
    elif average == 'macro':
        precisions = []
        classes = np.unique(y_true)
        for clss in classes:
            tp = np.sum((y_true == clss) & (y_pred == clss))
            fp = np.sum((y_true != clss) & (y_pred == clss))
            precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0
            precisions.append(precision_cls)
        return np.mean(precisions)


def recall_score(y_true, y_pred, average='binary'):
    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if tp + fn > 0 else 0
    elif average == 'macro':
        recalls = []
        classes = np.unique(y_true)
        for clss in classes:
            tp = np.sum((y_true == clss) & (y_pred == clss))
            fn = np.sum((y_true == clss) & (y_pred != clss))
            recall_cls = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(recall_cls)
        return np.mean(recalls)


def f1_score(y_true, y_pred, average='binary'):
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


def f1b_score(y_true, y_pred, b=1, average='binary'):
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    return (1 + b * b) * precision * recall / (b * b * precision + recall) if (b * b * precision + recall) > 0 else 0
