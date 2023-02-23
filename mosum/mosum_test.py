import numpy as np

def asymptoticA(x: float):
  """Help function: asymptotic scaling."""
  return np.sqrt(2*np.log(x))


def asymptoticB(x: float, K:int):
  """Help function: asymptotic shift"""
  return 2*np.log(x) + 0.5*np.log(np.log(x)) + np.log((K**2+K+1)/(K+1)) - 0.5*np.log(np.pi)



def criticalValue(n, G_left, G_right, alpha):
  """Computes the asymptotic critical value for the MOSUM test"""
  G_min = min(G_left, G_right)
  G_max = max(G_left, G_right)
  K = G_min / G_max
  return (asymptoticB(n/G_min,K) - np.log(np.log(1/np.sqrt(1-alpha))))/asymptoticA(n/G_min)

def pValue(z, n, G_left, G_right=None):
  """Computes the asymptotic p-value for the MOSUM test"""
  if G_right is None: G_right = G_left
  G_min = min(G_left, G_right)
  G_max = max(G_left, G_right)
  K = G_min / G_max
  return 1-np.exp(-2*np.exp(asymptoticB(n/G_min,K) - asymptoticA(n/G_min)*z))

