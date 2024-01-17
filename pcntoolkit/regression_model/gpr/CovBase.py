
from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
from scipy import optimize
from numpy.linalg import solve, LinAlgError
from numpy.linalg import cholesky as chol
from six import with_metaclass
from abc import ABCMeta, abstractmethod

try:  # Run as a package if installed
    from pcntoolkit.util.utils import squared_dist
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.dirname(path)  # parent directory
    if path not in sys.path:
        sys.path.append(path)
    del path

    from util.utils import squared_dist 


class CovBase(with_metaclass(ABCMeta)):
    """ Base class for covariance functions.

        All covariance functions must define the following methods::

            CovFunction.get_n_params()
            CovFunction.cov()
            CovFunction.xcov()
            CovFunction.dcov()
    """

    def __init__(self, x=None):
        self.n_params = np.nan

    def get_n_params(self):
        """ Report the number of parameters required """

        assert not np.isnan(self.n_params), \
            "Covariance function not initialised"

        return self.n_params

    @abstractmethod
    def cov(self, theta, x, z=None):
        """ Return the full covariance (or cross-covariance if z is given) """

    @abstractmethod
    def dcov(self, theta, x, i):
        """ Return the derivative of the covariance function with respect to
            the i-th hyperparameter """