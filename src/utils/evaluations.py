"""
Contains functions for evaluating the performance of the model.
Most functions take in the abundance, adjacency matrix, and a metric function, and return the score.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
)
from rpy2 import robjects
from rpy2.robjects import r, default_converter, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2 import rinterface
from utils.compositional_lotka_volterra import CompositionalLotkaVolterra
from utils.generalized_lotka_volterra import GeneralizedLotkaVolterra
from typing import Callable, List, Union


converter_context = (
    default_converter + numpy2ri.converter + pandas2ri.converter
).context()


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


def correlation_score(abundance, adj, method="pearson", **kwargs):
    if not isinstance(abundance, pd.DataFrame):
        abundance = pd.DataFrame(abundance)
    cor = abundance.corr(method)
    cor = cor.values
    return calc_nondiag_score(cor, adj, **kwargs)


correlation_score._method = "Pearson"


def spearman_score(abundance, adj, **kwargs):
    return correlation_score(abundance, adj, method="spearman", **kwargs)


spearman_score._method = "Spearman"


def precision_matrix_score(abundance, adj, **kwargs):
    if not isinstance(abundance, pd.DataFrame):
        abundance = pd.DataFrame(abundance)
    cor = abundance.cov().to_numpy()
    prec = np.linalg.inv(cor)
    return calc_nondiag_score(prec, adj, **kwargs)


precision_matrix_score._method = "Precision Matrix"


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


clv_score._method = "cLV"


def glv_score(abundance, adj, **kwargs):
    glv = GeneralizedLotkaVolterra([abundance], [np.arange(abundance.shape[0])])
    glv.set_regularizers(1e-3, 1e-3, 1e-3, 1e-3)
    glv.train()
    A, g, B = glv.get_params()
    return calc_nondiag_score(A, adj, **kwargs)


glv_score._method = "gLV"


def pcor_score(abundance, adj, **kwargs):
    ppcor = importr("ppcor")
    with converter_context:
        network_pred_ppea = ppcor.pcor(abundance)["estimate"]
    return calc_nondiag_score(network_pred_ppea, adj, **kwargs)


pcor_score._method = "Partial Pearson"


def pspe_score(abundance, adj, **kwargs):
    ppcor = importr("ppcor")
    with converter_context:
        network_pred_ppea = ppcor.pcor(abundance, method='spearman')["estimate"]
    return calc_nondiag_score(network_pred_ppea, adj, **kwargs)


pspe_score._method = "Partial Spearman"


def sparcc_score(abundance, adj, **kwargs):
    SpiecEasi = importr("SpiecEasi")
    with converter_context:
        network_pred_sparcc = SpiecEasi.sparcc(abundance)["Cor"]
    return calc_nondiag_score(network_pred_sparcc, adj, **kwargs)


sparcc_score._method = "SparCC"


def speic_score(abundance, adj, **kwargs):
    with converter_context:
        robjects.globalenv["abundance"] = abundance
        rcode = 'network_pred_speic = as.matrix(getOptMerge(SpiecEasi::spiec.easi(as.matrix(abundance), method="mb")))'
        r(rcode)
    network_pred_speic = np.array(robjects.globalenv["network_pred_speic"])
    return calc_nondiag_score(network_pred_speic, adj, **kwargs)


speic_score._method = "SpiecEasi"


def cclasso_score(abundance, adj, **kwargs):
    abundance = abundance.T
    NetCoMi = importr("NetCoMi")
    with converter_context:
        network_pred_cclasso = NetCoMi.cclasso(abundance, counts=True, pseudo=0.001)
    return calc_nondiag_score(network_pred_cclasso, adj, **kwargs)


cclasso_score._method = "CCLasso"


def baseline_score(abundance, adj, **kwargs):
    n = abundance.shape[1]
    pred = np.random.rand(n, n)
    return calc_nondiag_score(pred, adj, **kwargs)


baseline_score._method = "Baseline"


import torch
from neuralnet import AttentionNet, DNN
# model = AttentionNet(10, 500, 1024, 10).to('cuda')
model = DNN(50, 500, 1024).to('cuda')
model.load_state_dict(torch.load('d:\\microbial_network\\microbial_network_explore\\data\\attention_model.pt'))
def dl_score(abundance, adj, **kwargs):
    abundance = abundance.T
    abundance = torch.from_numpy(abundance).float().to('cuda')
    abundance = abundance[None, :, :]
    # adj_pred= model.calc_QK(abundance)[0].detach().cpu().numpy().squeeze()
    adj_pred = model(abundance).detach().cpu().numpy().squeeze()
    return calc_nondiag_score(adj_pred, adj, **kwargs)


dl_score._method = "Attention"
