import numpy as np
from sklearn.preprocessing import normalize

def clr(X):
    """Centered log ratio transformation of a matrix.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input matrix.
    Returns
    -------
    X_new : array-like, shape (n_samples, n_features)
        The transformed data.
    """
    X = np.asarray(X, dtype=np.float64)
    X = normalize(X, norm='l1', axis=1)
    X = np.log(X)
    X -= X.mean(axis=1)[:, np.newaxis]
    return X