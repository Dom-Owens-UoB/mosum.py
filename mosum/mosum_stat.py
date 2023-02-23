from typing import Union, Any

import numpy as np
import math
import sys
import pandas as pd

import warnings

class mosum_stat_obj:
    """mosum statistic"""
    def __init__(self, x, G_left, G_right, var_est_method, var_custom, boundary_extension, res, unscaledStatistic, var_estimation):
        self.x = x
        self.G_left = G_left
        self.G_right = G_right
        self.var_est_method = var_est_method
        self.var_custom = var_custom
        self.boundary_extension = boundary_extension
        self.stat = res
        self.rollsums = unscaledStatistic
        self.var_estimation = var_estimation

def mosum_stat(x, G, G_right=float("nan"), var_est_method='mosum', var_custom=None, boundary_extension=True) -> mosum_stat_obj:
    """ calculate mosum statistic """
    n = x.shape[0]

    # bandwidth checks
    symmetric = np.isnan(G_right)
    abs_bandwidth: Union[bool, Any] = (G >= 1)
    if not abs_bandwidth:
        G = int(np.floor(n * G))
        if not symmetric: G_right = int(np.floor(n * G_right))
    if (G < 1) & (G >= 0.5): sys.exit("Please use relative bandwidth between 0 and 0.5.")
    if G >= n / 2: sys.exit("Please use bandwidth smaller than n/2.")
    if not symmetric:
        if (G_right < 1) & (G_right >= 0.5): sys.exit("Please use relative bandwidth between 0 and 0.5.")
        if (G_right >= n / 2): sys.exit("Please use bandwidth smaller than n/2.")

    #consistency checks on input
    if len(x.shape) > 1: sys.exit(1)

    #bandwidths
    G_left = G
    if symmetric: G_right = G
    G_min = min(G_right, G_left)
    G_max = max(G_right, G_left)
    K = G_min / G_max

    #calculate stats
    sums_left = rolling_sum(x, G_left)# np.array(pd.Series(x).rolling(G_left).sum()[G_left:])
    if G_left == G_right:
        sums_right = sums_left
    else:
        sums_right = rolling_sum(x, G_right)#np.array(pd.Series(x).rolling(G_right).sum()[G_right:] )

    unscaledStatistic = np.concatenate((np.full(G_left - 1, float("nan")),
        (G_min / G_right * sums_right[G_left:] - G_min / G_left * sums_left[:n - G_left]) / (np.sqrt((K + 1) * G_min)),
        float("nan")), axis=None)
    # Calculate variance estimation.
    if (not var_custom is None) & (var_est_method != 'custom'): sys.exit('Please use var_est_method = custom when parsing var_custom.')
    if var_est_method == 'custom':
        if var_custom is None: sys.exit('Expecting var_custom to be not None for var_est_method=custom')
        if (len(var_custom) != n): sys.exit('Expecting var_custom to be of length n = len(x)')
        var = var_custom
    else:
    #if (var_est_method == 'global'):
    # Note: This is Deprecated
    #var = rep((sum(x ** 2) - (sum(x) ** 2) / n) / n, n)
    #else: # MOSUM-based variance estimators
        summedSquares_left = rolling_sum(x ** 2, G_left) #np.array(pd.Series(x**2).rolling(G_left).sum()[G_left:])
        squaredSums_left = sums_left ** 2
        var_tmp_left = summedSquares_left[:n - G_left+1] - 1 / G_left * (squaredSums_left[:n - G_left+1])
        var_left = np.concatenate( (np.full(G_left-1, float("nan")), var_tmp_left), axis=None) / G_left
        if G_left == G_right:
            summedSquares_right = summedSquares_left
            squaredSums_right = squaredSums_left
            var_tmp_right = var_tmp_left
        else:
            summedSquares_right =  rolling_sum(x ** 2, G_right) #np.array(pd.Series(x**2).rolling(G_right).sum()[G_right:])
            squaredSums_right = sums_right ** 2
            var_tmp_right = summedSquares_right[:n - G_right+1] - 1 / G_right * (squaredSums_right[:n - G_right+1])
        var_right = np.concatenate( (var_tmp_right[1: n - G_right +2], np.full(G_right, float("nan"))), axis=None ) / G_right #shifted
        if var_est_method == 'mosum':
            var = (var_left + var_right) / 2
        elif var_est_method == 'mosum_left':
        # Note: This is Deprecated
            var = var_left
        elif var_est_method == 'mosum_right':
            # Note: This is Deprecated
            var = var_right
        elif var_est_method == 'mosum_min':
            var = np.minimum(var_left, var_right)
        elif var_est_method == 'mosum_max':
            var = np.maximum(var_left, var_right)
        else: sys.exit('unknown variance estimation method')
    var_estimation = var

    # CUSUM extension to boundary
    if boundary_extension:
        if (n > 2 * G_left) :
            weights_left = np.sqrt( (G_left+G_right) / np.arange(1,G_left+1) / np.flip(np.arange(G_right, G_left + G_right)))
            unscaledStatistic[:G_left] = np.cumsum(np.mean(x[:G_left + G_right]) - x[:G_left]) * weights_left
            var_estimation[:G_left] = var_estimation[G_left-1]
        if n > 2 * G_right:
            with warnings.catch_warnings(): # zero-div handling
                warnings.simplefilter("ignore")
                weights_right = np.sqrt( (G_left+G_right) / np.flip(np.arange(0,G_right)) / (np.arange(G_left + 1, G_left + G_right+1)) )
                xrev: float = x[n - G_left - G_right:]
                unscaledStatistic[n - G_right:] = np.delete(np.cumsum(np.mean(xrev) - xrev),np.arange(0,G_left)) * weights_right
            unscaledStatistic[n-1] = 0
            var_estimation[n - G_right:] = var_estimation[n - G_right-1]
    res = np.absolute(unscaledStatistic) / np.sqrt(var_estimation)
    out = mosum_stat_obj(x, G_left, G_right, var_est_method, var_custom, boundary_extension, res, unscaledStatistic, var_estimation)
    return out




def rolling_sum(a: float, G: int = 1) -> np.ndarray: #replace this for speed-up?
    """ calculate rolling sums for mosum detector"""
    csum = np.cumsum(a, dtype=float)
    n = csum.size
    out = np.full(n, float("nan"))
    out[1:n-G+1] = csum[G:] - csum[:-G]
    out[0] = csum[G-1]
    return out  #[n - 1:]


def eta_criterion_help(candidates: int, m_values: float, eta: float, G_left: int, G_right: int):
    """eta localisation"""
    n = len(m_values)
    res = np.array([])
    left_length = int(np.floor(eta*G_left))
    right_length = int(np.floor(eta*G_right))
    for jj in np.arange(len(candidates)):
        k_star = candidates[jj];
        m_star = m_values[k_star];
        left_thresh = np.concatenate((0, k_star-left_length), axis=None).max()
        right_thresh = np.concatenate((n, k_star+right_length+1), axis=None).min()
        if m_star == max(m_values[left_thresh:right_thresh]):
            res = np.append(res, k_star)
    return res
