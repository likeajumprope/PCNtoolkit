import CovBase
import numpy as np
from utils import squaredist

class CovSqExpARD(CovBase):
    """ Squared exponential covariance function with ARD
        The hyperparameters are::

            theta = (log(ell_1, ..., log_ell_D), log(sf))

        where ell_i are lengthscale parameters and sf2 is the signal variance
    """

    def __init__(self, x=None):
        if x is None:
            raise ValueError("N x D data matrix must be supplied as input")
        if len(x.shape) == 1:
            self.D = 1
        else:
            self.D = x.shape[1]
        self.n_params = self.D + 1

    def cov(self, theta, x, z=None):
        self.ell = np.exp(theta[0:self.D])
        self.sf2 = np.exp(2*theta[self.D])

        if z is None:
            z = x

        R = squared_dist(x.dot(np.diag(1./self.ell)),
                         z.dot(np.diag(1./self.ell)))
        K = self.sf2*np.exp(-R/2)
        return K

    def dcov(self, theta, x, i):
        K = self.cov(theta, x)
        if i < self.D:    # return derivative of lengthscale parameter
            dK = K * squared_dist(x[:, i]/self.ell[i], x[:, i]/self.ell[i])
            return dK
        elif i == self.D:   # return derivative of signal variance parameter
            dK = 2*K
            return dK
        else:
            raise ValueError("Invalid covariance function parameter")
