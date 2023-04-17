#import pytest
import numpy as np
import mosum

n_test = 5
def test_big(): #"Test example detects all changes, matching R implementation
    x = np.hstack((mosum.testData("mix")["x"], mosum.testData("mix")["x"], mosum.testData("mix")["x"]))
    x = np.hstack((x,x,x,x,x))
    m_cpts = mosum.multiscale_localPrune(x, [10,30,50], criterion="epsilon").cpts
    assert len(m_cpts) > 0

def test_correct(): #"Test example detects all changes, matching R implementation
    x = mosum.testData("mix")["mu"] + np.tile([1.0,-1.0],280)
    m_cpts = mosum.multiscale_localPrune(x, [10,30], criterion="epsilon").cpts
    assert len(m_cpts) == len(mosum.testData("mix")["cpts"][0])
def test_single(): #"Single-bandwidth Multiscale CPTS bottom up equals MOSUM"
    sbic_str = -1*float('inf')
    for ii in range(n_test):
        ts = [mosum.testData("blocks"), mosum.testData("fms"), mosum.testData("mix"), mosum.testData("stairs10"),
              mosum.testData("teeth10")]
        alpha = float(np.random.uniform(0, 1, 1)[0])
        eps = float(np.random.uniform(0, 1, 1)[0])
        eta = float(np.random.uniform(0, 1, 1)[0])
        for ds in ts:
            x = ds["x"]
            n = len(x)
            G = int(np.random.uniform(5, 40, 1)[0])
            m_cpts = mosum.mosum(x, G, alpha=alpha, criterion="eta",eta=eta).cpts
            ml_cpts_pval = mosum.multiscale_localPrune(x, G, rule = "pval", alpha=alpha,
                                                       criterion="eta", eta=eta, pen_exp=sbic_str).cpts
            ml_cpts_peak = mosum.multiscale_localPrune(x, G, rule="jump", alpha=alpha,
                                                       criterion="eta", eta=eta, pen_exp=sbic_str).cpts
            if len(m_cpts) == 1:
                assert m_cpts == ml_cpts_pval
                assert m_cpts == ml_cpts_peak
            else:
                assert (m_cpts == ml_cpts_pval).all()
                assert (m_cpts == ml_cpts_peak).all()
            m_cpts = mosum.mosum(x, G, alpha=alpha, criterion="epsilon", epsilon=eps).cpts
            ml_cpts_pval = mosum.multiscale_localPrune(x, G, rule="pval", alpha=alpha,
                                                       criterion = "epsilon", epsilon=eps, pen_exp=sbic_str).cpts
            ml_cpts_peak = mosum.multiscale_localPrune(x, G, rule="jump", alpha=alpha,
                                                       criterion = "epsilon", epsilon=eps, pen_exp=sbic_str).cpts
            if len(m_cpts) == 1:
                assert m_cpts == ml_cpts_pval
                assert m_cpts == ml_cpts_peak
            else:
                assert (m_cpts == ml_cpts_pval).all()
                assert (m_cpts == ml_cpts_peak).all()


def test_inf(): #"Multiscale merging with infinite penalty yields empty set"
    sbic_str = float('inf')
    for ii in range(n_test):
        ts = [mosum.testData("blocks"), mosum.testData("fms"), mosum.testData("mix"), mosum.testData("stairs10"),
              mosum.testData("teeth10")]
        alpha = float(np.random.uniform(0, 1, 1)[0])
        eps = float(np.random.uniform(0, 1, 1)[0])
        eta = float(np.random.uniform(0, 1, 1)[0])
        for ds in ts:
            x = ds["x"]
            n = len(x)
            G = int(np.random.uniform(5, 40, 1)[0])
            ml_cpts_pval = mosum.multiscale_localPrune(x, G, rule = "pval", alpha=alpha,
                                                       criterion="eta", eta=eta, pen_exp=sbic_str).cpts
            ml_cpts_peak = mosum.multiscale_localPrune(x, G, rule="jump", alpha=alpha,
                                                       criterion="eta", eta=eta, pen_exp=sbic_str).cpts
            assert len(ml_cpts_pval) == 0
            assert len(ml_cpts_peak) == 0
            ml_cpts_pval = mosum.multiscale_localPrune(x, G, rule="pval", alpha=alpha,
                                                       criterion = "epsilon", epsilon=eps, pen_exp=sbic_str).cpts
            ml_cpts_peak = mosum.multiscale_localPrune(x, G, rule="jump", alpha=alpha,
                                                       criterion = "epsilon", epsilon=eps, pen_exp=sbic_str).cpts
            assert len(ml_cpts_pval) == 0
            assert len(ml_cpts_peak) == 0

def test_increase(): #"Increased penalty does not increase number of cpts"
    INF = float('inf')
    sbic_strengths = [-1*INF, 0, 1, 1.1, 1.5, 10, INF]
    for ii in range(n_test):
        ts = [mosum.testData("blocks"), mosum.testData("fms"), mosum.testData("mix"), mosum.testData("stairs10"),
              mosum.testData("teeth10")]
        alpha = float(np.random.uniform(0, 1, 1)[0])
        eps = float(np.random.uniform(0, 1, 1)[0])
        eta = float(np.random.uniform(0, 1, 1)[0])
        for ds in ts:
            x = ds["x"]
            n = len(x)
            G = int(np.random.uniform(5, 40, 1)[0])
            ml_eta_pval, ml_eta_peak, ml_eps_pval, ml_eps_peak = [ [] for _ in range(10)], [ [] for _ in range(10)], [ [] for _ in range(10)], [ [] for _ in range(10)] #[], [], [], []
            for jj in range(len(sbic_strengths)):
                sbic_str = sbic_strengths[jj]
                ml_eta_pval[jj] = mosum.multiscale_localPrune(x, G, rule = "pval", alpha=alpha,
                                                           criterion="eta", eta=eta, pen_exp=sbic_str).cpts
                ml_eta_peak[jj] = mosum.multiscale_localPrune(x, G, rule="jump", alpha=alpha,
                                                           criterion="eta", eta=eta, pen_exp=sbic_str).cpts
                ml_eps_pval[jj] = mosum.multiscale_localPrune(x, G, rule="pval", alpha=alpha,
                                                           criterion = "epsilon", epsilon=eps, pen_exp=sbic_str).cpts
                ml_eps_peak[jj] = mosum.multiscale_localPrune(x, G, rule="jump", alpha=alpha,
                                                           criterion = "epsilon", epsilon=eps, pen_exp=sbic_str).cpts
                if jj > 0:
                    print(jj)
                    assert len(ml_eta_pval[jj]) <= len(ml_eta_pval[jj-1])
                    assert len(ml_eta_peak[jj]) <= len(ml_eta_peak[jj-1])
                    assert len(ml_eps_pval[jj]) <= len(ml_eps_pval[jj-1])
                    assert len(ml_eps_peak[jj]) <= len(ml_eps_peak[jj-1])