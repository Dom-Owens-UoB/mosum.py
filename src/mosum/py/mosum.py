r"""MOSUM"""
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt

# from mosum_test import criticalValue
#from mosum_stat import mosum_stat
#from mosum_stat import eta_criterion_help
import mosum_stat as m_s
import mosum_test


class mosum_obj:
    """mosum object"""

    def __init__(self, x, G_left, G_right, var_est_method, var_custom, boundary_extension, stat, unscaledStatistic,
                 var_estimation,
                 threshold, alpha, threshold_custom, threshold_value, criterion, eta, epsilon, cpts, cpts_info,
                 do_confint, ci):
        """init method"""
        self.x = x
        self.G_left = G_left
        self.G_right = G_right
        self.var_est_method = var_est_method
        self.var_custom = var_custom
        self.boundary_extension = boundary_extension
        self.stat = stat
        self.rollsums = unscaledStatistic
        self.var_estimation = var_estimation
        self.threshold = threshold
        self.alpha = alpha
        self.threshold_custom = threshold_custom
        self.threshold_value = threshold_value
        self.criterion = criterion
        self.eta = eta
        self.epsilon = epsilon
        self.cpts = np.array(cpts, int)
        self.cpts_info = cpts_info
        self.do_confint = do_confint
        self.ci = ci  # note

    def plot(self, display=['data', 'mosum'][0], cpts_col='red', critical_value_col='blue', xlab='Time'):
        """plot method - plots data or detector"""
        plt.clf()
    #     if (is(self.x, 'ts'))
    #        x_plot <- as.numeric(time(self.x))
    #    elif(is(self.x, 'timeSeries'))
    #        x_plot <- time(self.x)
    #    else:
        x_plot = np.arange(0,len(self.x))
        if (display == 'mosum'):
            plt.plot(x_plot, self.stat, ls='-', color="black")
            plt.axhline(self.threshold_value, color=critical_value_col)
        if (display == 'data'):
            if(len(self.cpts)>0):
                brks = np.concatenate((0, self.cpts, len(self.x)), axis=None)
            else:
                brks = np.array([0, len(self.x)])
            fhat = self.x * 0
            for kk in np.arange(0,(len(brks) - 1)):
                int = np.arange(brks[kk],brks[kk + 1])
                fhat[int] = np.mean(self.x[int])
            plt.plot(x_plot, self.x, ls='-', color="black")
            plt.xlabel(xlab)
            plt.title("v")
            plt.plot(x_plot, fhat, color = 'darkgray', ls = '-', lw = 2)
        for p in self.cpts:
            plt.axvline(x_plot[np.int(p-1)]+1, color='red')


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
        out = 'change points detected with bandwidths (' + str(self.G_left) + ',' + str(
            self.G_right) + ') at alpha = ' + str(self.alpha) + ' according to ' + self.criterion + '-criterion'
        if (self.criterion == 'eta'): out = out + (' with eta = ' + str(self.eta))
        if (self.criterion == 'epsilon'): out = out + (' with epsilon = ' + str(self.epsilon))
        out = out + (' and ' + self.var_est_method + ' variance estimate:')
        print(out)
        if (len(self.cpts) > 0):
            print(ans)
        else:
            print('no change point is found')


def mosum(x, G, G_right=float("nan"), var_est_method=['mosum', 'mosum_min', 'mosum_max', 'custom'][0],
          var_custom=None, boundary_extension=True,
          threshold=['critical_value', 'custom'][0], alpha=.1,
          threshold_custom=float("nan"), criterion=['eta', 'epsilon'][0],
          eta=0.4, epsilon=0.2, do_confint=False,
          level=0.05, N_reps=1000):
    """
    MOSUM procedure for multiple change point estimation

    Computes the MOSUM detector, detects (multiple) change points and estimates their locations.

    Parameters
    ----------
    x : list
        input data
    G : int
        bandwidth; should be less than 'len(x)/2'
    G_right : int
        if 'G.right != G}, the asymmetric bandwidth '(G, G.right)' will be used;
        if 'max(G, G.right)/min(G, G.right) > 4', a warning message is generated
    var_est_method : how the variance is estimated; possible values are
        'mosum' : both-sided MOSUM variance estimator
        'mosum_min' : minimum of the sample variance estimates from the left and right summation windows
        'mosum_max' : maximum of the sample variance estimates from the left and right summation windows
        'custom' : a vector of 'len(x)' is to be parsed by the user; use 'var.custom' in this case to do so
    var_custom : float
        vector (of the same length as 'x}) containing local estimates of the variance or long run variance;
        use iff 'var.est.method = "custom"'
    boundary_extension : bool
        a logical value indicating whether the boundary values should be filled in with CUSUM values
    threshold : Str
        indicates which threshold should be used to determine significance.
        By default, it is chosen from the asymptotic distribution at the given significance level 'alpha`.
        Alternatively it is possible to parse a user-defined numerical value with 'threshold.custom'.
    alpha : float
        numeric value for the significance level with '0 <= alpha <= 1';
        use iff 'threshold = "critical_value"'
    threshold_custom : float
        value greater than 0 for the threshold of significance; use iff 'threshold = "custom"'
    criterion : Str
        indicates how to determine whether each point 'k' at which MOSUM statistic
        exceeds the threshold is a change point; possible values are
        'eta' : there is no larger exceeding in an 'eta*G' environment of 'k'
        'epsilon' : 'k' is the maximum of its local exceeding environment, which has at least size 'epsilon*G'
    eta : float
        a positive numeric value for the minimal mutual distance of changes,
        relative to moving sum bandwidth (iff 'criterion = "eta"')
    epsilon : float
        a numeric value in (0,1] for the minimal size of exceeding environments,
        relative to moving sum bandwidth (iff 'criterion = "epsilon"')
    do_confint : bool
        flag indicating whether to compute the confidence intervals for change points
    level : float
        use iff 'do_confint = True'; a numeric value ('0 <= level <= 1') with which '100(1-level)%'
        confidence interval is generated
    N_reps : int
        use iff 'do.confint = True'; number of bootstrap replicates to be generated

    Returns
    -------
    mosum_obj object containing
    x : list
        input data
    G_left, G_right : int
        bandwidths
    var_est_method, var_custom, boundary_extension : Str
        input
    stat : list
        MOSUM statistics
    rollsums : list
        MOSUM detector
    var_estimation : list
        local variance estimates
    threshold, alpha, threshold_custom
        input
    threshold_value : float
        threshold of MOSUM test
    criterion, eta, epsilon
        input
    cpts : ndarray
        estimated change point
    cpts_info : DataFrame
        information on change points, including detection bandwidths, asymptotic p-values, scaled jump sizes
    do_confint : bool
        input
    ci
        confidence intervals

    """
    # consistency checks on input

    n = x.shape[0]
    m = m_s.mosum_stat(x, G, G_right, var_est_method, var_custom, boundary_extension)

    G_left = m.G_left
    G_right = m.G_right
    G_min = min(G_right, G_left)
    G_max = max(G_right, G_left)
    K = G_min / G_max
    changePoints = np.array([])

    if threshold == 'critical_value':
        threshold_val = mosum_test.criticalValue(n, G_left, G_right, alpha)
    elif threshold == 'custom':
        threshold_val = threshold_custom
    else:
        sys.exit('threshold must be either critical_value or custom')
    # get exceeding TRUE/FALSE vector
    exceedings = m.stat > threshold_val

    # adjust, in case of no boundary CUSUM extension
    if not m.boundary_extension: exceedings[n - G_right] = False

    if criterion == 'epsilon':
        # get number of subsequent exceedings
        ex_len = pd.Series(exceedings)  # rlencode(exceedings)[1]
        exceedingsCount = np.array(exceedings * pd.Series(ex_len.groupby(ex_len).cumcount().add(1)))
        # get exceeding-intervals of fitting length
        minIntervalSize = max([1, (G_min + G_max) / 2 * epsilon])
        intervalEndPoints = np.array(np.where(np.diff(exceedingsCount) <= -minIntervalSize))
        intervalBeginPoints = intervalEndPoints - exceedingsCount[intervalEndPoints] + 1
        if not m.boundary_extension:
            # manually adjust right border
            if exceedings[n - G_right - 1] & (not (n - G_right) in intervalEndPoints):  # check all this
                lastBeginPoint = n - G_right - exceedingsCount[n - G_right] + 1
                if not (exceedings[seq(lastBeginPoint, n - G_right)]): sys.exit(0)
                if (lastBeginPoint in intervalBeginPoints): sys.exit(0)
                highestStatPoint = np.argmax(m.stat[lastBeginPoint:n - G_right]) + lastBeginPoint - 1
                if (highestStatPoint - lastBeginPoint >= minIntervalSize / 2):
                    intervalEndPoints = np.append(intervalEndPoints, n - G_right)
                    intervalBeginPoints = np.append(intervalBeginPoints, lastBeginPoint)
            # manually adjust left border
            if (exceedings[G_left] & (not G_left in intervalBeginPoints)):
                firstEndPoint = np.where(diff(exceedingsCount) < 0)[0]
                if not (exceedings[G_left:firstEndPoint]): sys.exit(0)
                if firstEndPoint in intervalEndPoints:
                    highestStatPoint = np.argmax(m.stat[G_left:firstEndPoint]) + G_left - 1
                if (firstEndPoint - highestStatPoint >= minIntervalSize / 2):
                    intervalEndPoints = np.insert(intervalEndPoints, 0, firstEndPoint)
                    intervalBeginPoints = np.insert(intervalBeginPoints, 0, G_left)

        numChangePoints = len(intervalBeginPoints)
        if (numChangePoints > 0):
            for ii in np.arange(numChangePoints):
                changePoint = intervalBeginPoints[ii] + np.argmax(
                    m.stat[np.arange(intervalBeginPoints[ii],intervalEndPoints[ii]+1)])  # - 1
                changePoints = np.append(changePoints, changePoint)
    elif (criterion=='eta'):
        localMaxima = np.logical_and(np.append(np.diff(m.stat)< 0, float("nan")) , np.append(float("nan"), np.diff(m.stat) > 0))
        # adjust, in case of no boundary CUSUM extension
        if not m.boundary_extension:
            localMaxima[n-G_right-1] = True
        p_candidates = np.where(exceedings & localMaxima)
        changePoints = m_s.eta_criterion_help(p_candidates, m.stat, eta, G_left, G_right)

    n_cps = len(changePoints)
    if n_cps == 0:
        cpts_info = None
    else:
        outcps = int(changePoints)
        cpts_info = pd.DataFrame({"cpts": outcps,
                                  "G_left": np.full(n_cps, G_left),
                                  "G_right": np.full(n_cps, G_right),
                                  "p_value": mosum_test.pValue(m.stat[outcps], n, G_left, G_right),
                                  "jump": np.sqrt((G_left + G_right) / G_left / G_right) * m.stat[outcps]})
    # if do_confint:
    #    ret$ci < - confint.mosum.cpts(ret, level=level, N_reps=N_reps)
    #    ret$do.confint < - TRUE
    ci = None

    out = mosum_obj(x, G_left, G_right, var_est_method, var_custom, boundary_extension, m.stat, m.rollsums,
                    m.var_estimation,
                    threshold, alpha, threshold_custom, threshold_val, criterion, eta, epsilon, changePoints, cpts_info,
                    do_confint, ci)
    return out
