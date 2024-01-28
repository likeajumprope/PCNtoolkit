import CovBase
import numpy as np
from utils import squaredist # fix this

class CovSqExp(CovBase):
    """ Ordinary squared exponential covariance function.
        The hyperparameters are::

            theta = ( log(ell), log(sf) )

        where ell is a lengthscale parameter and sf2 is the signal variance
    """

    def __init__(self, x=None):
        self.n_params = 2

    def cov(self, theta, x, z=None):
        self.ell = np.exp(theta[0])
        self.sf2 = np.exp(2*theta[1])

        if z is None:
            z = x

        R = squared_dist(x/self.ell, z/self.ell)
        K = self.sf2 * np.exp(-R/2)
        return K

    def dcov(self, theta, x, i):
        self.ell = np.exp(theta[0])
        self.sf2 = np.exp(2*theta[1])

        R = squared_dist(x/self.ell, x/self.ell)

        if i == 0:   # return derivative of lengthscale parameter
            dK = self.sf2 * np.exp(-R/2) * R
            return dK
        elif i == 1:   # return derivative of signal variance parameter
            dK = 2*self.sf2 * np.exp(-R/2)
            return dK
        else:
            raise ValueError("Invalid covariance function parameter")