import numpy as np
import mosum
xx = np.random.randn(100) + np.concatenate((np.full(50,0), np.full(50,4)))
xx_m = mosum.multiscale_bottomUp(xx)
xx_m.summary()
