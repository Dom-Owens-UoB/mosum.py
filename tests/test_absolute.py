#import pytest
import numpy as np
import mosum

n_test = 5
def test_symmetric(): #"MOSUM with relative and absolute symmetric bandwidths is consistent"
    for ii in range(n_test):
        n = int(np.floor(np.random.uniform(50, 1000, 1)))
        x = np.random.normal(0, 1, n)
        G_abs = 0
        while G_abs < 2:
            G_rel = np.random.uniform(0, 0.45, 1)
            G_abs = int(np.floor(n * G_rel))
        m_rel = mosum.mosum(x, G_rel)
        m_abs = mosum.mosum(x, G_abs)
        assert m_rel.G_left == m_abs.G_left
        assert (m_rel.stat == m_abs.stat).all()
def test_assymmetric(): #"MOSUM with relative and absolute assymmetric bandwidths is consistent"
    for ii in range(n_test):
        n = int(np.floor(np.random.uniform(50, 1000, 1)))
        x = np.random.normal(0, 1, n)
        G_left_abs = 0
        while G_left_abs < 2:
            G_left_rel = np.random.uniform(0, 0.45, 1)
            G_left_abs = int(np.floor(n * G_left_rel))
        G_right_abs = 0
        while G_right_abs < 2:
            G_right_rel = np.random.uniform(0, 0.45, 1)
            G_right_abs = int(np.floor(n * G_right_rel))
        m_rel = mosum.mosum(x, G_left_rel, G_right_rel)
        m_abs = mosum.mosum(x, G_left_abs, G_right_abs)
        assert m_rel.G_left == m_abs.G_left
        assert m_rel.G_right == m_abs.G_right
        assert (m_rel.stat == m_abs.stat).all()

def test_assymmetric_cpts(): #"MOSUM CPTS with relative and absolute assymmetric bandwidths is consistent"
    for ii in range(n_test):
        ts = [mosum.testData("blocks"), mosum.testData("fms"), mosum.testData("mix"), mosum.testData("stairs10"), mosum.testData("teeth10")]
        for ds in ts:
            x = ds["x"]
            n = len(x)
            G_left_abs = 0
            while G_left_abs < 2:
                G_left_rel = float(np.random.uniform(0, 0.45, 1)[0])
                G_left_abs = int(np.floor(n * G_left_rel))
            G_right_abs = 0
            while G_right_abs < 2:
                G_right_rel = float(np.random.uniform(0, 0.45, 1)[0])
                G_right_abs = int(np.floor(n * G_right_rel))
            m_rel = mosum.mosum(x, G_left_rel, G_right_rel)
            m_abs = mosum.mosum(x, G_left_abs, G_right_abs)
            assert (m_rel.cpts == m_abs.cpts).all()


def test_multiscale_cpts(): #"Multiscale MOSUM procedure with bottom-up merging with relative and absolute bandwidths is consistent"
    ts = [mosum.testData("blocks"), mosum.testData("fms"), mosum.testData("mix"), mosum.testData("stairs10"), mosum.testData("teeth10")]
    for ii in range(n_test):
        for ds in ts:
            x = ds["x"]
            n = len(x)
            G_left_abs = 0
            while G_left_abs < 2:
                G_left_rel = float(np.random.uniform(0, 0.45, 1)[0])
                G_left_abs = int(np.floor(n * G_left_rel))
            m_rel = mosum.multiscale_bottomUp(x, G_left_rel)
            m_abs = mosum.multiscale_bottomUp(x, G_left_abs)
            assert (m_rel.cpts == m_abs.cpts).all()
            assert (m_rel.pooled_cpts == m_abs.pooled_cpts).all()

