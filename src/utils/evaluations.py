"""
Contains functions for evaluating the performance of the model.
Most functions take in the abundance, adjacency matrix, and a metric function, and return the score.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
)
from rpy2.robjects import r, pandas2ri, default_converter, numpy2ri
from utils.compositional_lotka_volterra import CompositionalLotkaVolterra
from utils.generalized_lotka_volterra import GeneralizedLotkaVolterra
from typing import Callable, List, Union


def print_score(scores, metrics):
    if isinstance(scores, List):
        for metric, score in zip(metrics, scores):
            print(f"{metric.__name__}: {score}")
    else:
        print(f"{metrics.__name__}: {scores}")


def remove_diag(x):
    assert x.shape[0] == x.shape[1], "Input array must be square."
    return x[~np.eye(x.shape[0], dtype=bool)]


def calc_nondiag_score(
    prediction: np.ndarray, target: np.ndarray, metrics: Union[Callable, List[Callable]] = average_precision_score, verbose: bool = False
) -> float:
    """
    Calculates the score of a given prediction against a target using the specified metric. Only the non-diagonal elements are used.

    Parameters
    ----------
    prediction : ndarray of shape (n_vertices, n_vertices)
        The predicted values for the target.
    target : ndarray of shape (n_vertices, n_vertices)
        The true values for the target.
    metrics : callable, optional (default=average_precision_score)
        The metric function used to calculate the score. It should take in two
        arrays of the same shape and return a single value.

    Returns
    -------
    score : float
        The score of the prediction against the target.

    Notes
    -----
    - The metric function should return a higher value for better predictions.
    - The input arrays must have the same shape.

    """
    n_vertices = prediction.shape[0]
    idx = ~np.eye(n_vertices, dtype=bool)
    target = target[idx]
    prediction = prediction[idx]
    if isinstance(metrics, List):
        scores = [metric(target, prediction) for metric in metrics]
    else:
        scores = metrics(target, prediction)
    if verbose:
        print_score(scores, metrics)
    return scores


def correlation_score(abundance, adj, method="pearson", **kwargs):
    if not isinstance(abundance, pd.DataFrame):
        abundance = pd.DataFrame(abundance)
    cor = abundance.corr(method)
    cor = cor.values
    return calc_nondiag_score(cor, adj, **kwargs)


def precision_matrix_score(abundance, adj, **kwargs):
    if not isinstance(abundance, pd.DataFrame):
        abundance = pd.DataFrame(abundance)
    cor = abundance.cov().to_numpy()
    prec = np.linalg.inv(cor)
    return calc_nondiag_score(prec, adj, **kwargs)


def clv_score(abundance, adj, metrics=average_precision_score, verbose=False):
    clv = CompositionalLotkaVolterra([abundance], [np.arange(abundance.shape[0])])
    clv.set_regularizers(1e-3, 1e-3, 1e-3, 1e-3)
    clv.train()
    A, g, B = clv.get_params()
    denom = clv.denom
    rel_adj = np.delete(adj, denom, axis=0)
    if isinstance(metrics, List):
        scores = [metric(rel_adj.ravel(), A.ravel()) for metric in metrics]
    else:
        scores = metrics(rel_adj.ravel(), A.ravel())
    if verbose:
        print_score(scores, metrics)
    return scores


def glv_score(abundance, adj, **kwargs):
    glv = GeneralizedLotkaVolterra([abundance], [np.arange(abundance.shape[0])])
    glv.set_regularizers(1e-3, 1e-3, 1e-3, 1e-3)
    glv.train()
    A, g, B = glv.get_params()
    return calc_nondiag_score(A, adj, **kwargs)


def pcor_score(abundance, adj, **kwargs):
    if not isinstance(abundance, pd.DataFrame):
        abundance = pd.DataFrame(abundance)
    