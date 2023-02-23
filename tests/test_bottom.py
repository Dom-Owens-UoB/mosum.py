#import pytest
import numpy as np
import mosum

n_test = 5
def test_bottomup(): #"Single-bandwidth Multiscale CPTS bottom up equals MOSUM"
    for ii in range(n_test):
        ts = [mosum.testData("blocks"), mosum.testData("fms"), mosum.testData("mix"), mosum.testData("stairs10"),
              mosum.testData("teeth10")]
        alpha = float(np.random.uniform(0, 1, 1)[0])
        eta = float(np.random.uniform(0, 1, 1)[0])
        for ds in ts:
            x = ds["x"]
            n = len(x)
            G = int(max([np.random.uniform(20, 40, 1)[0], np.random.uniform(n/20, n/5, 1)[0]]))
            m_cpts = mosum.mosum(x, G, alpha=alpha, criterion="eta",eta=eta).cpts
            mb_cpts = mosum.multiscale_bottomUp(x, G, alpha=alpha, eta=eta).cpts
            if len(m_cpts) == 1:
                assert m_cpts == mb_cpts
            else:
                assert (m_cpts == mb_cpts).all()
