import warnings
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mosum.mosum import mosum
from mosum.mosum_test import pValue
from mosum.bandwidth import bandwidths_default, multiscale_grid

class multiscale_cpts:
    """multiscale_cpts object"""

    def __init__(self, x, cpts, cpts_info, pooled_cpts, G,
                 alpha, threshold, threshold_function, criterion, eta,
                 do_confint, ci):
        """init method"""
        self.x = x
        self.G = G
        self.threshold = threshold
        self.alpha = alpha
        self.threshold_function = threshold_function
        self.criterion = criterion
        self.eta = eta
        self.cpts = np.array(cpts, int)
        self.cpts_info = cpts_info
        self.pooled_cpts = pooled_cpts
        self.do_confint = do_confint
        self.ci = ci  # note
        self.var_est_method = "mosum"

    def plot(self, display=['data', 'mosum'][0], cpts_col='red', critical_value_col='blue', xlab='Time'):
        """plot method - plots data or detector"""
        plt.clf()
        x_plot = np.arange(0,len(self.x))
        if (display == 'mosum'):
            plt.plot(x_plot, self.stat, ls='-', color="black")
            plt.axhline(self.threshold_value, color=critical_value_col)
        if (display == 'data'):
            if(len(self.cpts)>0):
                brks = np.concatenate((0, self.cpts, len(self.x)), axis=None)
            else:
                brks = np.array([0, len(self.x)])
            brks.sort()
            fhat = self.x * 0
            for kk in np.arange(0,(len(brks) - 1)):
                int = np.arange(brks[kk],brks[kk + 1])
                fhat[int] = np.mean(self.x[int])
            plt.plot(x_plot, self.x, ls='-', color="black")
            plt.xlabel(xlab)
            plt.title("v")
            plt.plot(x_plot, fhat, color = 'darkgray', ls = '-', lw = 2)
        for p in self.cpts:
            plt.axvline(x_plot[p-1]+1, color='red')


    def summary(self):
        """summary method"""
        n = len(self.x)
        if (len(self.cpts) > 0):
            ans = self.cpts_info
            ans.p_value = round(ans.p_value, 3)
            ans.jump = round(ans.jump, 3)
        # if (self.do.confint): ans = pd.DataFrame(ans, self.ci$CI[, -1, drop=FALSE])

        #  cat(paste('created using mosum version ', utils::packageVersion('mosum'), sep=''))
        out = 'change points detected at alpha = ' + str(self.alpha) + ' according to ' + self.criterion + '-criterion'
        if (self.criterion == 'eta'): out = out + ' with eta = ' + str(self.eta)
        if (self.criterion == 'epsilon'): out = out + ' with epsilon = ' + str(self.epsilon)
        out = out + ' and ' + self.var_est_method + ' variance estimate:'
        print(out)
        if (len(self.cpts) > 0):
            print(ans)
        else:
            print('no change point is found')

    def print(self):
        """print method"""
        #  cat(paste('created using mosum version ', utils::packageVersion('mosum'), sep=''))
        n = len(self.x)
        if (len(self.cpts) > 0):
            ans = self.cpts_info
            ans.p_value = round(ans.p_value, 3)
            ans.jump = round(ans.jump, 3)
        out = 'change points detected with bandwidths (' + str(self.G) + ',' + str(
            self.G) + ') at alpha = ' + str(self.alpha) + ' according to ' + self.criterion + '-criterion'
        if (self.criterion == 'eta'): out = out + (' with eta = ' + str(self.eta))
        if (self.criterion == 'epsilon'): out = out + (' with epsilon = ' + str(self.epsilon))
        out = out + (' and ' + self.var_est_method + ' variance estimate:')
        print(out)
        if (len(self.cpts) > 0):
            print(ans)
        else:
            print('no change point is found')


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
    cpts : int
        estimated change points
    cpts_info : DataFrame
        information on change points
        self.pooled_cpts = pooled_cpts
        self.do_confint = do_confint
        self.ci = ci  # note
        self.var_est_method = "mosum"
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
    else:
        out = m
   # if (do.confint) {
   # ret$ci = confint.multiscale.cpts(ret, level=level, N_reps=N_reps)
   # ret$do.confint = TRUE

    return out

