import numpy as np
import sys
from test_models import modelSignal

def testData(model=['custom', 'blocks', 'fms', 'mix', 'stairs10', 'teeth10'][0], lengths= None, means=None, sds=None,
                                           rand_gen=np.random.normal, seed=None, rand_gen_args = [0,1]):
    """
    Test data with piecewise constant mean
    Generate piecewise stationary time series with independent innovations and change points in the mean.
    """
    np.random.seed(seed=seed)
    signal = testSignal(model=model, lengths=lengths, means=means, sds=sds)
    n = len(signal.mu_t)
    rand_gen_args = np.append(rand_gen_args, n)
    ts = signal.mu_t + rand_gen(*rand_gen_args)*signal.sigma_t
    out = {"x" : ts, "mu" : signal.mu_t, "sigma" : signal.sigma_t, "cpts" : np.where(np.diff(signal.mu_t) != 0)}
    return out

def testSignal(model=['custom', 'blocks', 'fms', 'mix', 'stairs10', 'teeth10'][0],
                                       lengths=None, means=None, sds=None):
  """
  Piecewise constant test signal

  Produce vectors of mean and dispersion values for generating piecewise stationary time series.
  :param model:
  :param lengths:
  :param means:
  :param sds:
  :return:
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
    res = [mu_t, sigma_t]
  elif model in ['blocks', 'fms', 'mix', 'stairs10', 'teeth10']:
      res = modelSignal(model) #test_models.
  else:
    sys.exit('Unknown model string')

  return(res)
