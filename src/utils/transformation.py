import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import eigvalsh

from rpy2 import robjects
from rpy2.robjects import r, default_converter, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2 import rinterface

converter_context = (default_converter + numpy2ri.converter + pandas2ri.converter).context()

r["source"]("src/utils/transformation.R")


def clr_transform(data):
    """
    Performs the centered log ratio (CLR) transformation on compositional data.

    Parameters
    ----------
    data : array-like
        A two-dimensional array of compositional data, where each row represents a sample and each column represents a component.

    Returns
    -------
    clr : ndarray
        An array of CLR-transformed values, with the same dimensions as the input data.
    """
    # Calculate the geometric mean of each row
    gm = np.exp(np.mean(np.log(data), axis=1, keepdims=True))

    # Calculate the CLR-transformed values
    # clr = np.log(data / gm)
    clr = data / gm

    return clr


def alr_transform(data, ref_component=0):
    """
    Performs the additive log ratio (ALR) transformation on compositional data, using the specified reference component.

    Parameters
    ----------
    data : array-like
        A two-dimensional array of compositional data, where each row represents a sample and each column represents a component.
    ref_component : int, optional
        The index of the reference component to use for the ALR transformation. Default is 0.

    Returns
    -------
    alr : ndarray
        An array of ALR-transformed values, with one fewer dimension than the input data.
    """
    # Select the reference component
    ref = data[:, ref_component]

    # Calculate the ALR-transformed values
    # alr = np.log(data[:, np.arange(data.shape[1]) != ref_component] / ref[:, np.newaxis])
    alr = data[:, np.arange(data.shape[1]) != ref_component] / ref[:, np.newaxis]

    return alr


def ilr_basis(n_components):
    """
    Computes the ILR basis matrix for a given number of components.

    Parameters
    ----------
    n_components : int
        The number of components in the compositional data.

    Returns
    -------
    ilr_basis : ndarray
        An (n_components-1) x (n_components-1) array representing the ILR basis matrix.
    """
    # Construct the centering matrix
    c = np.zeros((n_components, n_components - 1))
    for i in range(1, n_components):
        c[i, : i - 1] = 1 / np.sqrt(i * (i + 1))
        c[i, i - 1] = -i / np.sqrt(i * (i + 1))

    # Compute the ILR basis matrix
    ilr_basis = np.zeros((n_components - 1, n_components - 1))
    for i in range(n_components - 1):
        for j in range(i, n_components - 1):
            v = c[:, i : j + 1]
            ilr_basis[i, j] = np.sqrt(eigvalsh(v.T @ v)[-1])

    return ilr_basis


def ilr_transform(data):
    """
    Performs the isometric log ratio (ILR) transformation on compositional data.

    Parameters
    ----------
    data : array-like
        A two-dimensional array of compositional data, where each row represents a sample and each column represents a component.

    Returns
    -------
    ilr : ndarray
        An array of ILR-transformed values, with one fewer dimension than the input data.
    """
    # Compute the ILR basis matrix
    n_components = data.shape[1]
    basis = ilr_basis(n_components)

    # Compute the centered log ratio (CLR) values
    clr = np.log(data / np.exp(data.mean(axis=1, keepdims=True)))

    # Compute the ILR-transformed values
    ilr = clr @ basis

    return ilr


def css_transform(data):
    data = data.T
    with converter_context:
        css = r["css.func"](data)
    return css.to_numpy().T


css_transform._method = "CSS"


def tmm_transform(data):
    data = data.T
    with converter_context:
        tmm = r["tmm.func"](data)
    return tmm.to_numpy().T


tmm_transform._method = "TMM"


def tmmwsp_transform(data):
    data = data.T
    with converter_context:
        tmmwsp = r["tmmwsp.func"](data)
    return tmmwsp.to_numpy().T


tmmwsp_transform._method = "TMMwsp"


def rle_transform(data):
    data = data.T
    ## RLE only supports integer values
    data = (data * 1000).round()
    with converter_context:
        rle = r["rle.func"](data)
    return rle.to_numpy().T


rle_transform._method = "RLE"


def gmpr_transform(data):
    data = data.T
    with converter_context:
        gmpr = r["gmpr.func"](data)
    return gmpr.to_numpy().T

gmpr_transform._method = "GMPR"


def logcpm_transform(data):
    data = data.T
    with converter_context:
        logcpm = r["cpm"](data, log=True)
    return logcpm.to_numpy().T

logcpm_transform._method = "logCPM"

def ast_transform(data):
    data = np.arcsin(np.sqrt(data))
    return data

ast_transform._method = "AST"

def blom_transform(data):
    data = data.T
    with converter_context:
        blom = r["blom.func"](data)
    return blom.to_numpy().T

blom_transform._method = "Blom"

def vst_transform(data):
    data = data.T
    data = (data * 1000).round()
    with converter_context:
        vst = r["varianceStabilizingTransformation"](data)
    return vst.to_numpy().T

vst_transform._method = "VST"

def npn_transform(data):
    importr("huge")
    with converter_context:
        data = r["huge.npn"](data)
    return data

npn_transform._method = "NPN"


def qn_transform(ref, data):
    importr("preprocessCore")
    ref = ref.T
    data = data.T
    # log_ref = np.log(ref)
    # log_data = np.log(data)
    ref_quantiles = r['normalize.quantiles.determine.target'](x=robjects.r.matrix(robjects.FloatVector(log_ref)))
    with converter_context:
        qn = r['normalize.quantiles.use.target'](log_data, target=ref_quantiles)
    return qn.T

qn_transform._method = "QN"

def fsqn_trasform(ref, data):
    importr('FSQN')
    # log_ref = np.log(ref)
    # log_data = np.log(data)
    with converter_context:
        fsqn = r['quantileNormalizeByFeature'](matrix_to_normalize=log_data, target_distribution_matrix=log_ref)
    return fsqn

fsqn_trasform._method = "FSQN"

def bmc_transform(ref, data):
    importr('pamr')
    # log_ref = np.log(ref)
    # log_data = np.log(data)
    with converter_context:
        bmc = r['bmc'](log_data, log_ref)
    return bmc.to_numpy()[:, log_ref.shape[1]:].T

bmc_transform._method = "BMC"


def no_transform(data):
    return data


no_transform._method = "None"
