#import pytest
import numpy as np
import mosum



n_test = 5
n_bootstrap = 1000
num_tol = 1e-12

def test_boot(): #"Width of bootstrap intervals is consistent"
    for ii in range(n_test):
        ts = [mosum.testData("blocks"), mosum.testData("fms"), mosum.testData("mix"), mosum.testData("stairs10"),
              mosum.testData("teeth10")]
        alpha = float(np.random.uniform(0, 1, 1)[0])
        #eps = float(np.random.uniform(0, 1, 1)[0])
        #eta = float(np.random.uniform(0, 1, 1)[0])
        for ds in ts:
            x = ds["x"]
            n = len(x)
            G_left = max(int(np.random.uniform(20, 40, 1)[0]), int(np.random.uniform(0.05*n, .1*n, 1)[0]))
            G_right = max(int(np.random.uniform(20, 40, 1)[0]), int(np.random.uniform(0.05 * n, .1 * n, 1)[0]))
            m_cpts1 = mosum.mosum(x, G_left, G_right, alpha=alpha)
            m_cpts2 = mosum.multiscale_bottomUp(x, G=[G_left, G_right], alpha=alpha)
            m_cpts3 = mosum.multiscale_localPrune(x, G=[G_left, G_right], alpha=alpha)

            s1 = np.random.random_integers(1, 10000, 1)[0]
            s2 = np.random.random_integers(1, 10000, 1)[0]
            s3 = np.random.random_integers(1, 10000, 1)[0]

            alpha1 = np.random.uniform(0, .5, 1)[0]
            np.random.seed(s1)
            b1 = m_cpts1.confint(N_reps=n_bootstrap, level=alpha1)
            np.random.seed(s2)
            b2 = m_cpts2.confint(N_reps=n_bootstrap, level=alpha1)
            np.random.seed(s3)
            b3 = m_cpts3.confint(N_reps=n_bootstrap, level=alpha1)

            # Pointwise intervals contain estimate
            if len(b1["CI"]["cpts"])>0:
                assert all(b1["CI"]["pw.left"] <= b1["CI"]["cpts"]) and all(b1["CI"]["pw.right"] >= b1["CI"]["cpts"])
            if len(b2["CI"]["cpts"]) > 0:
                assert all(b2["CI"]["pw.left"] <= b2["CI"]["cpts"]) and all(b2["CI"]["pw.right"] >= b2["CI"]["cpts"])
            if len(b3["CI"]["cpts"])>0:
                assert all(b3["CI"]["pw.left"] <= b3["CI"]["cpts"]) and all(b3["CI"]["pw.right"] >= b3["CI"]["cpts"])

            # Uniform intervals contain estimate
            if len(b1["CI"]["cpts"]) > 0:
                assert all(b1["CI"]["unif.left"] - num_tol <= b1["CI"]["cpts"]) and all(b1["CI"]["unif.right"] + num_tol >= b1["CI"]["cpts"])
            if len(b2["CI"]["cpts"]) > 0:
                assert all(b2["CI"]["unif.left"] - num_tol <= b2["CI"]["cpts"]) and all(b2["CI"]["unif.right"] + num_tol >= b2["CI"]["cpts"])
            if len(b3["CI"]["cpts"]) > 0:
                assert all(b3["CI"]["unif.left"] - num_tol <= b3["CI"]["cpts"]) and all(b3["CI"]["unif.right"] + num_tol >= b3["CI"]["cpts"])

            ## comment out jit instances in `bootstrap.py` to test
            #alpha2 = np.random.uniform(0, alpha1, 1)[0]
            #seed(s1)
            #b4 = m_cpts1.confint(N_reps=n_bootstrap, level=alpha2)
            #seed(s2)
            # b5 = m_cpts2.confint(N_reps=n_bootstrap, level=alpha2)
            #seed(s3)
            #b6 = m_cpts3.confint(N_reps=n_bootstrap, level=alpha2)

            # Pointwise Intervals grow (with smaller alpha)
            #if len(b1["CI"]["cpts"]) > 0:
            #    assert all(b1["CI"]["pw.left"] >= b4["CI"]["pw.left"]) and all(b1["CI"]["pw.right"] <= b4["CI"]["pw.right"])
            #if len(b2["CI"]["cpts"]) > 0:
            #    assert all(b2["CI"]["pw.left"] >= b5["CI"]["pw.left"]) and all(b2["CI"]["pw.right"] <= b5["CI"]["pw.right"])
            #if len(b3["CI"]["cpts"]) > 0:
            #    assert all(b3["CI"]["pw.left"] >= b6["CI"]["pw.left"]) and all(b3["CI"]["pw.right"] <= b6["CI"]["pw.right"])

            # Uniform Intervals grow (with smaller alpha)
            #if len(b1["CI"]["cpts"]) > 0:
            #    assert all(b1["CI"]["unif.left"] + num_tol >= b4["CI"]["unif.left"]) and all(b1["CI"]["unif.right"]- num_tol <= b4["CI"]["unif.right"])
            #if len(b2["CI"]["cpts"]) > 0:
            #    assert all(b2["CI"]["unif.left"] + num_tol>= b5["CI"]["unif.left"]) and all(b2["CI"]["unif.right"]- num_tol <= b5["CI"]["unif.right"])
            #if len(b3["CI"]["cpts"]) > 0:
            #    assert all(b3["CI"]["unif.left"] + num_tol>= b6["CI"]["unif.left"]) and all(b3["CI"]["unif.right"]- num_tol <= b6["CI"]["unif.right"])
