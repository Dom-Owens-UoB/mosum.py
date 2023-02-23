import numpy as np
import mosum
import pytest

n_test = 5
def test_margins(): #"Asymmetric MOSUM bandwidth margins are consistent"
    for ii in range(n_test):
        n = int(2*np.floor(np.random.uniform(25, 500, 1))-1) #odd n
        x = np.random.normal(0, 1, n)
        with pytest.raises(SystemExit) as e_info:
            mosum.mosum(x, G = 0, G_right=10)
            mosum.mosum(x, G=10, G_right=0)
            mosum.mosum(x, G=n, G_right=10)
            mosum.mosum(x, G=10, G_right=n)
        #n/2
        GG = int(np.floor(n / 2))
        mn_half1 = mosum.mosum(x, G=GG, boundary_extension=False).stat
        assert np.sum(np.logical_not(np.isnan(mn_half1))) == 2
        with pytest.raises(SystemExit) as e_info:
            mn_half2 = mosum.mosum(x, G=GG+1, boundary_extension=False).stat
        #assert np.isnan(mn_half2).all()
        mn_half3 = mosum.mosum(x, G=GG, G_right=GG-1, boundary_extension=False).stat
        assert np.sum(np.logical_not(np.isnan(mn_half3))) == 3
        mn_half4 = mosum.mosum(x, G=GG-1, G_right=GG, boundary_extension=False).stat
        assert np.sum(np.logical_not(np.isnan(mn_half4))) == 3
        #random
        G_left = int(np.floor(np.random.uniform(5, n/8, 1)))
        G_right = int(np.floor(np.random.uniform(5, n / 8, 1)))
        m = mosum.mosum(x, G=G_left, G_right = G_right, boundary_extension=False).stat
        m2 = mosum.mosum(x, G=G_left, G_right=G_right, boundary_extension=True).stat
        margin = np.concatenate((np.arange(G_left-1), np.arange(n-G_right, n)), axis=None)
        inner = np.arange(G_left-1, n-G_right)
        assert (np.arange(n) == np.sort(np.concatenate((margin,inner), axis=None))).all()
        assert np.isnan(m[margin]).all()
        assert not np.isnan(m[inner]).any()
        assert np.sum(np.isnan(m2)) == 0

def test_margins(): #"Asymmetric MOSUM with symmetric bandwidth coincides with regular MOSUM"
    for ii in range(n_test):
        n = int(np.floor(np.random.uniform(50, 1000, 1)))
        x = np.random.normal(0, 1, n)
        G = int(np.floor(np.random.uniform(5, n/8, 1)))
        assert (mosum.mosum(x, G).stat == mosum.mosum(x, G, G_right=G).stat).all()