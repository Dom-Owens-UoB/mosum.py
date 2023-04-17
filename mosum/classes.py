import mosum
import numpy as np
from matplotlib import pyplot as plt

from mosum.bootstrap import confint_mosum_cpts, confint_multiscale_cpts



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
            plt.axvline(x_plot[(p-1)]+1, color=cpts_col)


    def summary(self):
        """summary method"""
        n = len(self.x)
        if (len(self.cpts) > 0):
            ans = self.cpts_info
            ans.p_value = round(ans.p_value, 3)
            ans.jump = round(ans.jump, 3)
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

    def confint(self, parm: str = "cpts", level: float = 0.05, N_reps: int = 1000):
        """
        Generate bootstrap confidence intervals for change points

        Parameters
        ----------
        parm : Str
            unused
        level : float
            numeric value in (0, 1), such that the `100(1-level)%` confidence bootstrap intervals are computed
        N_reps : int
            number of bootstrap replicates

        Returns
        -------
        dictionary containing inputs, pointwise intervals and uniform intervals
        """
        return confint_mosum_cpts(self, parm, level, N_reps)





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

    def confint(self, parm: str = "cpts", level: float = 0.05, N_reps: int = 1000):
        """
        Generate bootstrap confidence intervals for change points

        Parameters
        ----------
        parm : Str
            unused
        level : float
            numeric value in (0, 1), such that the `100(1-level)%` confidence bootstrap intervals are computed
        N_reps : int
            number of bootstrap replicates

        Returns
        -------
        dictionary containing inputs, pointwise intervals and uniform intervals
        """
        return confint_multiscale_cpts(self, parm, level, N_reps)


class multiscale_cpts_lp(multiscale_cpts):
    def __init__(self, x, cpts, cpts_info, pooled_cpts, G,
                 alpha, threshold, threshold_function, criterion, eta,
                 epsilon, sc, rule, penalty, pen_exp,
                 do_confint, ci):
        """init method"""
        self.x = x
        self.G = G
        self.threshold = threshold
        self.alpha = alpha
        self.threshold_function = threshold_function
        self.criterion = criterion
        self.eta = eta
        self.epsilon = epsilon
        self.sc = sc
        self.rule = rule
        self.penalty = penalty
        self.pen_exp = pen_exp
        self.cpts = np.array(cpts, int)
        self.cpts_info = cpts_info
        self.pooled_cpts = pooled_cpts
        self.do_confint = do_confint
        self.ci = ci  # note
        self.var_est_method = "mosum"

    def plot(self, display=['data', 'significance'][0], shaded =['CI', 'bandwidth', 'none'][0],
             level = 0.05, N_reps = 1000, CI = ['pw', 'unif'][0], xlab = "Time"):
        """plot method - plots data or p-values, shaded according to confidence intervals or detection bandwidth"""
        if shaded == 'bandwidth':
            main = 'Change point estimators and detection intervals'
        elif shaded == 'CI':
            if CI == 'pw':
                main = f"Change point estimators and pointwise {100 * (1 - level)}% confidence intervals"
            elif CI == 'unif':
                main = f"Change point estimators and uniform {100 * (1 - level)}% confidence intervals"
            if len(self.cpts) > 0:
                if self.do_confint:
                    b = self.ci
                else:
                    b = confint_multiscale_cpts(self, level=level, N_reps=N_reps)
        elif shaded == 'none':
            main = 'Change point estimators'
        else:
            raise ValueError("shaded argument has to be either 'CI', 'bandwidth' or 'none'.")

        n = len(self.x)
        x_plot = np.arange(0, n)
        q = len(self.cpts)
        if q > 0:
            cpts = self.cpts_info
            ls = np.linspace(0,1,q)
            cols = [plt.cm.prism(i, alpha=0.2) for i in ls]
            cols2 = [plt.cm.prism(i, alpha=1.0) for i in ls]


            xx = self.cpts  # location
            if shaded == 'bandwidth':
                xx_l = np.maximum(0, xx - cpts["G_left"] + 1)
                xx_r = np.minimum(n, xx + cpts["G_right"])
            elif shaded == 'CI':
                if CI == 'pw':
                    xx_l = b["CI"]["pw.left"].astype(int)
                    xx_r = b["CI"]["pw.right"].astype(int)
                elif CI == 'unif':
                    xx_l = b["CI"]["unif.left"].astype(int)
                    xx_r = b["CI"]["unif.right"].astype(int)

        if display == 'data':
            brks = np.concatenate(([0], self.cpts, [n]))
            fhat = np.zeros(n)
            for kk in range(len(brks) - 1):
                intv = np.arange(brks[kk], brks[kk + 1])
                fhat[intv] = np.mean(self.x[intv])
            plt.plot(x_plot, self.x, 'b-', label='data', color = "black")
            plt.plot(x_plot, fhat, 'k-', label='mean', color = "darkgrey")
            if q > 0:
                plt.vlines(x_plot[xx], ymin=plt.ylim()[0], ymax=plt.ylim()[1], colors=cols2, linestyles='dashed')
                if shaded != 'none':
                    for kk in range(q):
                        plt.fill_between(x_plot[xx_l[kk]:xx_r[kk]], y1=plt.ylim()[0], y2=plt.ylim()[1], facecolor=cols[kk], alpha=0.2)
        if display == 'significance':
            y_min = max(1 - 1.1 * self.alpha, 0)
            plt.plot(0)
            plt.xlim((0,n))
            plt.ylim((y_min,1))
            plt.xlabel(xlab)
            plt.ylabel('1 - p_value')
            plt.title(main)
            if q > 0:
                yy = 1 - cpts["p_value"]  # pvalue
                plt.scatter(x_plot[xx], yy)
                plt.vlines(x_plot[xx], ymin=np.zeros(len(yy)), ymax=yy, colors=cols2)
                #plt.plot([x_plot[xx], x_plot[xx]], np.vstack([np.zeros(len(yy)), yy]), color = cols2)#, col=cols2
                if shaded != 'none':
                    for kk in range(q):
                        plt.fill_between(x_plot[xx_l[kk]:xx_r[kk]], 0, yy[kk], facecolor=cols[kk], linestyle='-', linewidth=0)
