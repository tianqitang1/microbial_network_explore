import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import eigvalsh

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
        c[i, :i-1] = 1 / np.sqrt(i*(i+1))
        c[i, i-1] = -i / np.sqrt(i*(i+1))
    
    # Compute the ILR basis matrix
    ilr_basis = np.zeros((n_components - 1, n_components - 1))
    for i in range(n_components - 1):
        for j in range(i, n_components - 1):
            v = c[:, i:j+1]
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
