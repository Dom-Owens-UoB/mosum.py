#import pytest
import numpy as np
import mosum

n_test = 20
def test_threshold(): #"Custom threshold is consistent with asymptotic critical value"
    for ii in range(n_test):
        ts = [mosum.testData("blocks"), mosum.testData("fms"), mosum.testData("mix"), mosum.testData("stairs10"), mosum.testData("teeth10")]
        alpha = float(np.random.uniform(0, 1, 1)[0])
        for ds in ts:
            x = ds["x"]
            n = len(x)
            G_left = int(max([np.random.uniform(20, 40, 1)[0], np.random.uniform(n/20, n/5, 1)[0]]))
            G_right = int(max([np.random.uniform(20, 40, 1)[0], np.random.uniform(n / 20, n / 5, 1)[0]]))

            th_custom = mosum.criticalValue(n, G_left, G_right, alpha)
            def th_1(G,n,alpha): return mosum.criticalValue(n, G, G, alpha)
            #def th_2(G_l, G_r, n, alpha): mosum.criticalValue(n, G_l, G_r, alpha)


            m_cpts = mosum.mosum(x, G_left, G_right, alpha=alpha).cpts
            mb_cpts = mosum.multiscale_bottomUp(x, G = [G_left, G_right], alpha=alpha).cpts
            #mlp
            m_custom_cpts = mosum.mosum(x, G_left, G_right, alpha=alpha, threshold="custom", threshold_custom=th_custom).cpts
            m_th1_cpts = mosum.multiscale_bottomUp(x, G = [G_left, G_right], alpha=alpha, threshold="custom", threshold_function=th_1).cpts
            #m_th2_cpts = mosum.mosum(x, G_left, G_right, alpha=alpha, threshold="custom", threshold_function=th_2).cpts

            assert (m_cpts == m_custom_cpts).all()
            assert (mb_cpts == m_th1_cpts).all()
            #assert (mb_cpts == m_th1_cpts).all()