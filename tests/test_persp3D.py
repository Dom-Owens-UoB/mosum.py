import mosum.persp3D
import numpy as np
import mosum
from matplotlib import pyplot as plt
xx = np.random.randn(100) + np.concatenate((np.full(50,0), np.full(50,4)))
mosum.persp3D.persp3D_multiscaleMosum(xx)