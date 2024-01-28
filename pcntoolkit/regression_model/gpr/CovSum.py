import CovBase
import CovLin
import CovSqExpARD


class CovSum(CovBase):
    """ Sum of covariance functions. These are passed in as a cell array and
        intialised automatically. For example::

            C = CovSum(x,(CovLin, CovSqExpARD))
            C = CovSum.cov(x, )

        The hyperparameters are::

            theta = ( log(ell_1, ..., log_ell_D), log(sf2) )

        where ell_i are lengthscale parameters and sf2 is the signal variance
    """

    def __init__(self, x=None, covfuncnames=None):
        if x is None:
            raise ValueError("N x D data matrix must be supplied as input")
        if covfuncnames is None:
            raise ValueError("A list of covariance functions is required")
        self.covfuncs = []
        self.n_params = 0
        for cname in covfuncnames:
            covfunc = eval(cname + '(x)')
            self.n_params += covfunc.get_n_params()
            self.covfuncs.append(covfunc)

        if len(x.shape) == 1:
            self.N = len(x)
            self.D = 1
        else:
            self.N, self.D = x.shape

    def cov(self, theta, x, z=None):
        theta_offset = 0
        for ci, covfunc in enumerate(self.covfuncs):
            try:
                n_params_c = covfunc.get_n_params()
                theta_c = [theta[c] for c in
                           range(theta_offset, theta_offset + n_params_c)]
                theta_offset += n_params_c
            except Exception as e:
                print(e)

            if ci == 0:
                K = covfunc.cov(theta_c, x, z)
            else:
                K += covfunc.cov(theta_c, x, z)
        return K

    def dcov(self, theta, x, i):
        theta_offset = 0
        for covfunc in self.covfuncs:
            n_params_c = covfunc.get_n_params()
            theta_c = [theta[c] for c in
                       range(theta_offset, theta_offset + n_params_c)]
            theta_offset += n_params_c

            if theta_c:  # does the variable have any hyperparameters?
                if 'dK' not in locals():
                    dK = covfunc.dcov(theta_c, x, i)
                else:
                    dK += covfunc.dcov(theta_c, x, i)
        return dK