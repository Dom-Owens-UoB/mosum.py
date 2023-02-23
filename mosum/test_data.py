import numpy as np
import sys
from mosum.test_models import modelSignal

def testData(model=['custom', 'blocks', 'fms', 'mix', 'stairs10', 'teeth10'][1], lengths= None, means=None, sds=None,
                                           rand_gen=np.random.normal, seed=None, rand_gen_args = [0,1]):
    """
    Test data with piecewise constant mean

    Generate piecewise stationary time series with independent innovations and change points in the mean.

    Parameters
    ----------
    model : str
        custom or pre-defined signal
    lengths : int
        vector of segment lengths (`custom` only)
    means : int
        vector of segment means (`custom` only)
    sds : int
        vector of segment standard deviations (`custom` only)
    rand_gen : function
        innovation function
    seed : int
        random seed
    rand_gen_args : ndarray
        arguments for `rand_gen`

    Returns
    -------
    x : ndarray
        simulated data series
    mu : ndarray
        signal
    sigma : float
        standard deviation
    cpts : ndarray
        true change points

    Examples
    --------
    >>> mosum.testData()
    >>> mosum.testData("custom", lengths = [100,100], means=[0,1], sds= [1,1])
    """
    np.random.seed(seed=seed)
    signal = testSignal(model=model, lengths=lengths, means=means, sds=sds)
    n = len(signal["mu_t"])
    rand_gen_args = np.append(rand_gen_args, n)
    ts = signal["mu_t"] + rand_gen(*rand_gen_args)*signal["sigma_t"]
    out = {"x" : ts, "mu" : signal["mu_t"], "sigma" : signal["sigma_t"], "cpts" : np.where(np.diff(signal["mu_t"]) != 0)}
    return out

def testSignal(model=['custom', 'blocks', 'fms', 'mix', 'stairs10', 'teeth10'][1],
                                       lengths=None, means=None, sds=None):
  """
  Piecewise constant test signal

  Produce vectors of mean and dispersion values for generating piecewise stationary time series.
  """
  if (model=='custom'):
    if lengths is None: sys.exit()
    if means is None: sys.exit()
    if sds is None: sys.exit()
    if not len(lengths) == len(means): sys.exit()
    if not len(lengths) == len(sds): sys.exit()
    mu_t = np.array([])
    sigma_t = np.array([])
    for ii in np.arange(len(lengths)):
      mu_t = np.append(mu_t, np.full(lengths[ii], means[ii]))
      sigma_t = np.append(sigma_t, np.full(lengths[ii], sds[ii]))
    res = {'mu_t' : mu_t, 'sigma_t' : sigma_t}
  elif model in ['blocks', 'fms', 'mix', 'stairs10', 'teeth10']:
      res = modelSignal(model) #test_models.
  else:
    sys.exit('Unknown model string')

  return(res)
