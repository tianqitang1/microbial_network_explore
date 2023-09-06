import contextlib
import io
import sys
import numpy as np
from typing import Callable, List, Union
from sklearn.metrics import average_precision_score

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout


def print_score(scores, metrics):
    if isinstance(scores, List):
        for metric, score in zip(metrics, scores):
            print(f"{metric.__name__}: {score}")
    else:
        print(f"{metrics.__name__}: {scores}")


def calc_nondiag_score(
    prediction: np.ndarray,
    target: np.ndarray,
    metrics: Union[Callable, List[Callable]] = average_precision_score,
    rowwise: bool = False,
    verbose: bool = False,
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
    rowwise : bool, optional (default=False)
        Whether to calculate the score row-wise. If True, the metric function
        will be applied to each row of the prediction and target arrays.

    Returns
    -------
    score : float
        The score of the prediction against the target.

    Notes
    -----
    - The metric function should return a higher value for better predictions.
    - The input arrays must have the same shape.

    """
    if rowwise:
        if isinstance(metrics, List):
            scores = [np.mean([metric(target[i, np.arange(target.shape[1])!=i], prediction[i, np.arange(target.shape[1])!=i])
            for i in range(target.shape[0])])
            for metric in metrics]
        else:
            scores = np.mean([metrics(target[i, np.arange(target.shape[1])!=i], prediction[i, np.arange(target.shape[1])!=i])
            for i in range(target.shape[0])])
        
    else:
        n_vertices = prediction.shape[0]
        idx = ~np.eye(n_vertices, dtype=bool)
        target = target[idx]
        prediction = np.abs(prediction[idx])
        if isinstance(metrics, List):
            scores = [metric(target, np.nan_to_num(prediction)) for metric in metrics]
        else:
            scores = metrics(target, np.nan_to_num(prediction))
    if verbose:
        print_score(scores, metrics)
    return scores