import numpy as np
from sklearn.preprocessing import normalize

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
    clr = np.log(data / gm)
    
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
    alr = np.log(data[:, np.arange(data.shape[1]) != ref_component] / ref[:, np.newaxis])
    
    return alr