import sys
import numpy as np

def bandwidths_default(n, d_min=10, G_min=10, G_max=None) -> int:
  """Default choice for the set of multiple bandwidths"""
  if G_max is None: G_max = min(n / 2, n ** (2 / 3))
  G_0 = G_1 = max(G_min, round(2/3*d_min))
  G = [G_0, G_1]
  j = 2
  G_j = G[j-2] + G[j-1]
  while (G_j <= G_max):
    G = np.append(G, G_j)
    j = j+1
    G_j = G[j-2] + G[j-1]
  return G[1:]

class multiscale_grid_obj:
  """multiscale_grid object"""
  def __init__(self, grid, max_imbalance):
    """init method"""
    self.grid = grid
    self.max_imbalance = max_imbalance


def multiscale_grid(bandwidths_left, bandwidths_right=None,
                            method='cartesian', max_unbalance=4):
  """Multiscale bandwidth grids"""
  if bandwidths_right is None: bandwidths_right = bandwidths_left
  if not all(np.array(bandwidths_left) > 0): sys.exit()
  if not all(np.array(bandwidths_right) > 0): sys.exit()
  if not max_unbalance >= 1.0: sys.exit()
  H_left = []
  H_right = []
  if (method == 'cartesian'):
    for G_left in bandwidths_left:
      for G_right in bandwidths_right:
        ratio = max(G_left, G_right) / min(G_left, G_right)
        if (ratio <= max_unbalance):
          H_left = np.append(H_left, G_left)
          H_right = np.append(H_right, G_right)
  else:
    if not method == 'concatenate': sys.exit()
    if not (len(bandwidths_left)==len(bandwidths_right)): sys.exit()
    H_left = bandwidths_left
    H_right = bandwidths_right
  if not (len(H_left)==len(H_right)): sys.exit()
  out = multiscale_grid_obj([H_left, H_right], max_unbalance)
  return out

