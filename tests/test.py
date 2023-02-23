import numpy as np
import mosum
from matplotlib import pyplot as plt
xx = np.random.randn(100) + np.concatenate((np.full(50,0), np.full(50,4)))
#xx_ms = mosum_stat(xx, G = 15)
criterion = ["eta","epsilon"][0]
xx_m  = mosum.mosum(xx, G = 15, criterion = criterion, boundary_extension = False)
xx_m.summary()
xx_m.print()
#xx_m.plot()
xx_m.plot(display="mosum")
#ts = pd.Series(xx_ms.stat)
#ts.plot()
plt.show()

