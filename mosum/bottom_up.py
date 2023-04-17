import warnings
import sys
import numpy as np
import pandas as pd

from mosum.mosum import mosum
from mosum.mosum_test import pValue
from mosum.bandwidth import bandwidths_default, multiscale_grid
from mosum.classes import multiscale_cpts
from mosum.bootstrap import confint_multiscale_cpts, confint_mosum_cpts


def multiscale_bottomUp(x, G=None, threshold = ['critical_value', 'custom'][0],
                    alpha = 0.1, threshold_function = None, eta = 0.4, do_confint = False, level = 0.05, N_reps = 1000):
    """
    Multiscale MOSUM algorithm with bottom-up merging

    Parameters
    ----------
    x : list
        input data
    G : int
        vector of bandwidths; given as either integers less than `len(x)/2`,
         or numbers between `0` and `0.5` describing the moving sum bandwidths relative to `len(x)`
    threshold : Str
        indicates which threshold should be used to determine significance.
        By default, it is chosen from the asymptotic distribution at the given significance level 'alpha`.
        Alternatively it is possible to parse a user-defined function with 'threshold_function'.
    alpha : float
        numeric value for the significance level with '0 <= alpha <= 1';
        use iff 'threshold = "critical_value"'
    threshold_function : function
    eta : float
        a positive numeric value for the minimal mutual distance of changes,
        relative to moving sum bandwidth (iff 'criterion = "eta"')
    do_confint : bool
        flag indicating whether to compute the confidence intervals for change points
    level : float
        use iff 'do_confint = True'; a numeric value ('0 <= level <= 1') with which '100(1-level)%'
        confidence interval is generated
    N_reps : int
        use iff 'do.confint = True'; number of bootstrap replicates to be generated

    Returns
    -------
    multiscale_cpts object containing
    x : list
        input data
    G : int
        bandwidth vector
    threshold, alpha, threshold_function, eta
        input
    cpts : ndarray
        estimated change point
    cpts_info : DataFrame
        information on change points, including detection bandwidths, asymptotic p-values, scaled jump sizes
    pooled_cpts : ndarray
        change point candidates
    do_confint : bool
        input
    ci
        confidence intervals

    Examples
    --------
    >>> import mosum
    >>> xx = mosum.testData("blocks")["x"]
    >>> xx_m  = mosum.multiscale_bottomUp(xx, G = [50,100])
    >>> xx_m.summary()
    >>> xx_m.print()
    """
    n = len(x)
    if G is None:
        G = bandwidths_default(n, G_min=max(20, np.ceil(0.05 * n)))
        grid = multiscale_grid(G, method='concatenate')
    elif type(G) in [int, float]:
        grid = multiscale_grid([G], method='concatenate')
    elif type(G) == 'multiscale_grid_obj':
        if any(G.grid[1] - G.grid[0] != 0): sys.exit("Expecting a grid of symmetric bandwidths")
        grid = G
    elif type(G) == list:
        G.sort()
        grid = multiscale_grid(G, method='concatenate')
    else: sys.exit('Expecting a vector of numbers')
    abs_bandwidth = (np.array(grid.grid) >= 1).all()

    if abs_bandwidth:
        GRID_THRESH = max([20, 0.05 * n])
    else:
        GRID_THRESH = 0.05

    if (threshold == 'critical_value') & (min(grid.grid[0]) < GRID_THRESH):
        warnings.warn('Smallest bandwidth in grid is relatively small (in comparison to n), \n increase the smallest bandwidth or use multiscale.localPrune instead')

    if (not threshold == 'critical_value') & (not threshold == 'custom'):
        warnings.warn('threshold must be either \'critical.value\' or \'custom\'')


    if not (alpha >= 0) & (alpha <= 1): sys.exit("alpha out of range")
    if not (eta <= 1) & (eta > 0): sys.exit("eta out of range")
    if not (not do_confint or N_reps > 0): sys.exit()
    
    # Retreive change point candidates from all bandwidths.
    cpts_complete = []
    bandwidths_complete = []
    pValues_complete = []
    jumps_complete = []

    GG = len(grid.grid[0])
    for i in range(GG):
        G = grid.grid[0][i]
        if threshold == 'critical_value':
            m = mosum(x, G, threshold='critical_value', alpha=alpha, criterion='eta', eta=eta)
        else:
            threshold_val = threshold_function(G, n, alpha)
            m = mosum(x, G, threshold='custom', threshold_custom=threshold_val, alpha=alpha, criterion='eta', eta=eta)
        if not abs_bandwidth:
            G = int(np.floor(G * n))
        if GG >= 2:
            cpts = m.cpts
            cpts_complete = np.append(cpts_complete, cpts)
            bandwidths_complete = np.append(bandwidths_complete, np.full(len(cpts), G))
            pValues_complete = np.append(pValues_complete, pValue(m.stat[cpts], n, G))
            jumps_complete = np.append(jumps_complete, m.stat[cpts] * np.sqrt(2 / G))
    
    # Merge candidates.
    if GG >= 2:
        points = [0]
        bandwidths = []
        pValues = []
        jumps = []
        cptsInOrder = range(len(cpts_complete))
        for i in cptsInOrder:
            p = cpts_complete[i]
            G = bandwidths_complete[i]
            pVal = pValues_complete[i]
            jmp = jumps_complete[i]
            #print(np.abs(p-points))
            if min(np.concatenate((np.abs(p-points), float("inf")),axis=None)) >= eta * G:  # Note: min(empty_list) = Inf
                points = np.append(points, p)
                bandwidths = np.append(bandwidths, G)
                pValues = np.append(pValues, pVal)
                jumps = np.append(jumps, jmp)
        cpts_merged = pd.DataFrame({"cpts" : points[1:], "G_left" : bandwidths, "G_right" : bandwidths,
        "p_value" : pValues, "jump" : jumps})
        cpts = cpts_merged["cpts"][cpts_merged.cpts.argsort().argsort()]
        G = grid.grid[0]
        if not abs_bandwidth:
            G = np.floor(n * G)
        out = multiscale_cpts(x,cpts, cpts_merged, np.sort(np.unique(cpts_complete)),G,
                          alpha,threshold, threshold_function, 'eta', eta,
                          False, None)  # note
        if do_confint:
            out.ci = confint_multiscale_cpts(out, level=level, N_reps=N_reps)
            out.do_confint = True
    else:
        out = m
        if do_confint:
            out.ci = confint_mosum_cpts(out, level=level, N_reps=N_reps)
            out.do_confint = True
    return out
