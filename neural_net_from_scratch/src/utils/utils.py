import numpy as np


def augment_features( X:np.ndarray) -> np.ndarray:
        return np.hstack([X, np.ones((X.shape[0], 1))])

def one_hot( y:np.ndarray, n_classes:int) -> np.ndarray:
    n_samples = y.shape[0]
    y_one_hot = np.zeros((n_samples, n_classes))
    y_one_hot[np.arange(n_samples), y] = 1
    return y_one_hot