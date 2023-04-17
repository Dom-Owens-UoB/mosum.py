import mosum.persp3D
import numpy as np
import mosum
from matplotlib import pyplot as plt
xx = np.random.randn(100) + np.concatenate((np.full(50,0), np.full(50,4)))
mosum.persp3D.persp3D_multiscaleMosum(xx)

import mosum
import mosum.local_prune
import numpy as np
from matplotlib import pyplot as plt
xx = np.random.randn(200) + np.concatenate((np.full(50,0), np.full(50,4), np.full(50,0), np.full(50,4)))
m = mosum.multiscale_localPrune(xx, [10,30], criterion="epsilon")
m.plot(shaded="bandwidth")
plt.show()
m.plot(shaded="bandwidth", display="significance")
plt.show()

