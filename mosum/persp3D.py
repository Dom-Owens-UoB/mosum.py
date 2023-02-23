import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

import mosum.bandwidth


def persp3D_multiscaleMosum(x, mosum_args = dict(), threshold = ['critical_value', 'custom'][0],
                            alpha = 0.1, threshold_function = None, palette = cm.coolwarm, xlab = "G", ylab = "time", zlab = "MOSUM"):
    """
    3D Visualisation of multiscale MOSUM statistics

    Parameters
    ----------
    x : list
        input data
    mosum_args : dict
        dictionary of keyword arguments to `mosum`
    threshold : Str
        indicates which threshold should be used to determine significance.
        By default, it is chosen from the asymptotic distribution at the given significance level 'alpha`.
        Alternatively it is possible to parse a user-defined function with 'threshold_function'.
    alpha : float
        numeric value for the significance level with '0 <= alpha <= 1';
        use iff 'threshold = "critical_value"'
    threshold_function : function
    palette : matplotlib.colors.LinearSegmentedColormap
        colour palette for plotting, accessible from `matplotlib.cm`
    xlab, ylab, zlab : Str
        axis labels for plot

    Examples
    --------
    >>> import mosum
    >>> xx = mosum.testData("blocks")["x"]
    >>> mosum.persp3D_multiscaleMosum(xx)
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    n = len(x)
    G = np.arange(8, int(np.floor(3*np.sqrt(n)))) + 1
    grid = mosum.bandwidth.multiscale_grid(G, method="concatenate")

    # collect stats for vis
    mrange = range(len(grid.grid[0]))
    m = list(mrange)
    for ii in mrange:
        G = grid.grid[0][ii]
        arglist = mosum_args
        arglist["x"] = x
        arglist["G"] = G
        if threshold=="custom":
            arglist["threshold"] = "custom"
            arglist["threshold_custom"] = threshold_function(G,n,alpha)
        m[ii] = mosum.mosum(**arglist)

    zz = [(lambda z: z.stat / z.threshold_value)(z) for z in m]
    xx = grid.grid[0]
    yy = np.arange(n)

    # Plot the surface.
    for ii in mrange:
        ax.scatter(np.full(n, xx[ii]),yy,zz[ii], c = palette(np.abs(zz[ii]/zz[ii].max())), edgecolor="none" )
        ax.plot(np.full(n, xx[ii]),yy,zz[ii], c = "lightgrey")
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)

    plt.show()

