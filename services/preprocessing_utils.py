import numpy as np

def safe_log1p_array(X):
    X = np.asarray(X)
    X = np.where(X < -1, 0, X)
    return np.log1p(X)
