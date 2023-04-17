"""Moving Sum Based Procedures for Changes in the Mean"""

from .mosum import mosum
#from .classes import mosum_obj, multiscale_cpts, multiscale_cpts_lp
from .mosum_test import criticalValue
from .bottom_up import multiscale_bottomUp
from .local_prune import multiscale_localPrune
from .bandwidth import bandwidths_default
from .test_data import testData
from .persp3D import persp3D_multiscaleMosum
#from .bootstrap import cpts_bootstrap
