#import pytest
import numpy as np
import mosum

n_test = 5
def test_epsilon(): #"Set of detected change points increases with decreasing epsilon"
    for ii in range(n_test):
        ts = [mosum.testData("blocks"), mosum.testData("fms"), mosum.testData("mix"), mosum.testData("stairs10"), mosum.testData("teeth10")]
        alpha = float(np.random.uniform(0, 1, 1)[0])
        for ds in ts:
            x = ds["x"]
            cpts_prev = []
            cpts_prev_b = []
            G_left = int(np.random.uniform(5, 15, 1)[0])
            ran = np.arange(1,12)
            ran = ran[::-1]/10
            for epsilon in ran:
                m_cur = mosum.mosum(x, G_left, boundary_extension=False, alpha=alpha, criterion="epsilon",
                                    epsilon=epsilon)
                m_cur_b = mosum.mosum(x, G_left, boundary_extension=True, alpha=alpha, criterion="epsilon",
                                    epsilon=epsilon)
                assert (set(cpts_prev) <= set(m_cur.cpts)) #(cpts_prev in m_cur.cpts).all()
                assert (set(cpts_prev_b) <= set(m_cur_b.cpts)) #(cpts_prev_b in m_cur_b.cpts).all()
                cpts_prev = m_cur.cpts
                cpts_prev_b = m_cur_b.cpts

def test_eta(): #"Set of detected change points increases with decreasing epsilon"
    for ii in range(n_test):
        ts = [mosum.testData("blocks"), mosum.testData("fms"), mosum.testData("mix"), mosum.testData("stairs10"),
              mosum.testData("teeth10")]
        alpha = float(np.random.uniform(0, 1, 1)[0])
        for ds in ts:
            x = ds["x"]
            cpts_prev = []
            cpts_prev_b = []
            G_left = int(np.random.uniform(5, 15, 1)[0])
            ran = np.arange(2,12)
            ran = ran[::-1]/10
            for eta in ran:
                m_cur = mosum.mosum(x, G_left, boundary_extension=False, alpha=alpha, criterion="eta",
                                    eta=eta)
                m_cur_b = mosum.mosum(x, G_left, boundary_extension=True, alpha=alpha, criterion="eta",
                                    eta=eta)
                assert (set(cpts_prev) <= set(m_cur.cpts)) #(cpts_prev in m_cur.cpts).all()
                assert (set(cpts_prev_b) <= set(m_cur_b.cpts)) #(cpts_prev_b in m_cur_b.cpts).all()
                cpts_prev = m_cur.cpts
                cpts_prev_b = m_cur_b.cpts

