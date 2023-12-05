# Import libraries
import numpy as np
import math
from modules.utils import *


def ED_distance(ts1, ts2):
    """
    Calculate the Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    ed_dist : float
        Euclidean distance between ts1 and ts2.
    """
    
    # Define variable-totalizer
    ed_dist = 0

    # Check lenght's of time serieses
    if len(ts1) != len(ts2):
        raise ValueError("Arrays must have the same length")

    # Count dinstances by formula
    for i,j in zip(ts1, ts2):
        ed_dist += (i-j)**2

    return ed_dist**(0.5)


def norm_ED_distance(ts1, ts2):
    """
    Calculate the normalized Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    norm_ed_dist : float
        The normalized Euclidean distance between ts1 and ts2.
    """
    # Check lenght's of time serieses
    if len(ts1) != len(ts2):
        raise ValueError("Arrays must have the same length")

    n = len(ts1)

    # Definition of indicators
    ev1 = sum(ts1)/n
    ev2 = sum(ts2)/n
    sd1 = np.sqrt(sum(ts1**2 - ev1**2) / n)
    sd2 = np.sqrt(sum(ts2**2 - ev2**2) / n)

    # Count complex_fraction by formula
    complex_fraction = 1 - ((np.dot(ts1,ts2)-n*ev1*ev2)/(n*sd1*sd2))

    return (abs(2*n*(complex_fraction)))**0.5


def DTW_distance(ts1, ts2, r=None):
    """
    Calculate DTW distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    r : float
        Warping window size.

    Returns
    -------
    dtw_dist : float
        DTW distance between ts1 and ts2.
    """

    n = len(ts1)
    m = len(ts2)

    # Initialize distance matrix
    DTW = np.zeros((n+1, m+1))
    DTW[:, :] = np.inf
    DTW[0, 0] = 0

    # Compute distance matrix 
    for i in range(1, n + 1):
        if (r is None):
            for j in range(1, n + 1):
                cost = (ts1[i-1] - ts2[j-1])**2
                DTW[i, j] = cost + min(DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1])
        else:
            left = max(1, i - int(np.floor(m*r)))
            right = min(m, i + int(np.floor(m*r))) + 1
            for j in range(left, right):
                cost = (ts1[i-1] - ts2[j-1])**2
                DTW[i, j] = cost + min(DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1])

    return DTW[n, m]


def DTW_matrix(ts1, ts2):
    """
    Function for printing DWT_matrix 

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.
    
    Returns
    -------
    DTW : float
        DTW matrix
    """ 
    n = len(ts1) + 1
    m = len(ts2) + 1
    DTW = np.zeros((n, m))
    DTW[:, 0] = float('Inf')
    DTW[0, :] = float('Inf')
    DTW[0, 0] = 0

    for i in range(1, n):
        for j in range(1, m):
            cost = (ts1[i-1] - ts2[j-1])**2
            DTW[i, j] = cost + min(DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1])

    return DTW


def calculate_distance_matrix(data, metric='euclidean', normalize=False):
    """
    Calculate distance matrix

    Parameters
    ----------
    data : numpy.ndarray
        Dataset

    metric : 'euclidean', 'dtw'
        User defined metric
    
    Returns
    -------
    DTW : numpy.ndarray
        Distance matrix
    """ 
    
    # Define amount of time series
    N = data.shape[0]
    # Initialize the distance matrix
    distance_matrix = np.zeros(shape=(N, N))
    # Define data
    in_data = data.copy()
    
    if metric=='euclidean':
        if normalize:
            dist_func = norm_ED_distance
        else:
            dist_func = ED_distance
    elif metric=='dtw':
        if normalize:
            for i in range(N):
                data[i] = z_normalize(data[i])
        dist_func = DTW_distance
    else:
        raise ValueError("Incorrect input metric")

    for i in range(N):
        for j in range(i, N):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                distance_matrix[i, j] = dist_func(data[i], data[j])

    # Define a lower-triangular matrix by transposition
    distance_matrix = distance_matrix + distance_matrix.T

    return distance_matrix