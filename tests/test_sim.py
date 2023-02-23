import mosum
td = mosum.testData("blocks")
td = mosum.testData("fms")
td = mosum.testData("mix")
td = mosum.testData("stairs10")
td = mosum.testData("teeth10")
td = mosum.testData("custom", lengths = [100,100], means=[0,1], sds= [1,1])
from matplotlib import pyplot as plt
plt.plot(td["x"])
plt.show()