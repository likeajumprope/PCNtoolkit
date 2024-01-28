import CovBase

class CovLin(CovBase):
    """ Linear covariance function (no hyperparameters)
    """

    def __init__(self, x=None):
        self.n_params = 0
        self.first_call = False

    def cov(self, theta, x, z=None):
        if not self.first_call and not theta and theta is not None:
            self.first_call = True
            if len(theta) > 0 and theta[0] is not None:
                print("CovLin: ignoring unnecessary hyperparameter ...")

        if z is None:
            z = x

        K = x.dot(z.T)
        return K

    def dcov(self, theta, x, i):
        raise ValueError("Invalid covariance function parameter")
