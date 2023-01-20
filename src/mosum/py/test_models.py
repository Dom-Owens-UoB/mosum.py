import numpy as np
def modelSignal(model=['custom', 'blocks', 'fms', 'mix', 'stairs10', 'teeth10'][0]):
  if (model=='blocks'):
      signal_means = [0, 14.64, -3.66, 7.32, -7.32,
                             10.98, -4.39, 3.29, 19.03, 7.68, 15.37, 0]
      signal_Cpts = [204, 266, 307, 471, 511, 819, 901, 1331, 1556, 1597, 1658, 2048]
      signal_sigma = 10
  elif (model=='fms'):
      signal_means =[-0.18, 0.08, 1.07, -0.53, 0.16, -0.69, -0.16]
      signal_Cpts = [138, 225, 243, 299, 308, 332, 497]
      signal_sigma = 0.3
  elif (model=='mix'):
      signal_means =[7, -7, 6, -6, 5, -5, 4, -4, 3, -3, 2, -2, 1, -1]
      signal_Cpts = [10, 20, 40, 60, 90, 120, 160, 200, 250, 300, 360, 420, 490, 560]
      signal_sigma = 4
  elif (model=='stairs10'):
      signal_means =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
      signal_Cpts =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
      signal_sigma = 0.3
  elif (model=='teeth10'):
      signal_means =[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
      signal_Cpts =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
      signal_sigma = 0.4
  signal_lengths = np.append(signal_Cpts[0], np.diff(signal_Cpts))

  mu_t = np.array([])
  for ii in np.arange(len(signal_lengths)):
      mu_t = np.append(mu_t, np.full(signal_lengths[ii],signal_means[ii] ))
  sigma_t = np.full(len(mu_t), signal_sigma)
  out = {'mu_t' : mu_t, 'sigma_t' : sigma_t}
  return out


 