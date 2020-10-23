"""Module for Kernel Management"""

import numpy as np
from scipy.spatial import distance_matrix
from scipy.special import gamma, kv

class Kernel():
    """Class to implement Kernel Functions"""

    def __init__(self):
        """Initialize Hyperparameters"""
        pass

    def compute_covariance(self, X, X_star = np.array([])):
        """Compute Covariance Matrix"""
        pass

    def set_values(self, values):
        """Return a kernel of the same form with its hyperparameters
        corresponding to the values given, if the kernel is composed, the set_values
        are distributed according to the depth first exploration of the tree"""
        pass

    def get_values(self):
        """Return the parameters of the kernel
        Returns:
            list of parameters
        """
        pass


class Add_Kernel(Kernel):
    def __init__(self, kernel1, kernel2):
        super().__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.n_params = self.kernel1.n_params + self.kernel2.n_params

    def compute_covariance(self, X, X_star = np.array([])):
        """Compute Covariance Matrix"""
        return self.kernel1.compute_covariance(X, X_star) + self.kernel2.compute_covariance(X, X_star)

    def set_values(self, values):
        """Compute Updates kernels and return their sum"""

        n1 = self.kernel1.n_params
        kernel1 = self.kernel1.set_values(values[:n1])
        kernel2 = self.kernel2.set_values(values[n1:])
        return Add_Kernel(kernel1, kernel2)

    def get_values(self):
        """Return the parameters of the kernel"""

        return self.kernel1.get_values() + self.kernel2.get_values()

class Multiply_Kernel(Kernel):
    def __init__(self, kernel1, kernel2):
        super().__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.n_params = self.kernel1.n_params + self.kernel2.n_params


    def compute_covariance(self, X, X_star = np.array([])):
        """Compute Covariance Matrix"""
        return self.kernel1.compute_covariance(X, X_star) * self.kernel2.compute_covariance(X, X_star)

    def set_values(self, values):
        """Compute Updates kernels and return their product"""

        n1 = self.kernel1.n_params
        kernel1 = self.kernel1.set_values(values[:n1])
        kernel2 = self.kernel2.set_values(values[n1:])
        return Multiply_Kernel(kernel1, kernel2)

    def get_values(self):
        """Return the parameters of the kernel"""

        return self.kernel1.get_values() + self.kernel2.get_values()

class RBF(Kernel):
    """Instantiate a RBF Kernel
    Args:
        l : lengthscale
    """

    def __init__(self, l = 1., var = 1.):
        super().__init__()
        self.lengthscale = l
        self.var = var
        self.n_params = 2

    def compute_covariance(self, X, X_star = np.array([])):
        """Compute Covariance Matrix for the RBF kernel K(X,X) or K(X, X_star)

        Args:
            X (nxp numpy matrix) : Data on which to compute
            X_star (optional, n x p numpy matrix) : Second dataset
        """
        if X_star.shape[0]:
            K_ = distance_matrix(X, X_star)
        else:
            K_ = distance_matrix(X, X)

        K_ =  np.exp(- K_ ** 2 / (2*self.lengthscale**2))

        return (self.var**2)*K_

    def set_values(self, values):
        """Compute Updates kernel for RBF"""

        kernel = RBF(values[0], values[1])
        return kernel

    def get_values(self):
        """Return the parameters of the kernel"""

        return [self.lengthscale, self.var]

class Periodic(Kernel):
    """Instantiate a Periodic Kernel

    Args:
        l : lengthscale
        p : period
        var : variance

    """

    def __init__(self, l = 1., p = 1., var = 1.):
        super().__init__()
        self.lengthscale = l
        self.period = p
        self.var = var
        self.n_params = 3


    def compute_covariance(self, X, X_star = np.array([])):
        """Compute Covariance Matrix for the Periodic kernel K(X,X) or K(X, X_star)

        Args:
            X (n x p numpy matrix) : Data on which to compute
            X_star (optional, n x p numpy matrix) : Second dataset
        """

        if X_star.shape[0]:
            K_ = distance_matrix(X, X_star, p = 1)
        else:
            K_ = distance_matrix(X, X, p = 1)

        K_ =  np.exp((-2/(self.lengthscale**2))*(np.sin(np.pi * K_ / self.period)**2))

        return (self.var**2)*K_

    def set_values(self, values):
        """Compute Updates kernel for Periodic Kernel"""

        kernel = Periodic(values[0], values[1], values[2])
        return kernel

    def get_values(self):
        """Return the parameters of the kernel"""

        return [self.lengthscale, self.period, self.var]

class Matern(Kernel):
    """Instantiate a Matern Kernel

    Args:
        l : lengthscale
        p : period
        var : variance

    """

    def __init__(self, l = 1., mu = 1/2, var = 1.):
        super().__init__()
        self.lengthscale = l
        self.mu = mu
        self.var = var
        self.n_params = 3

    def compute_covariance(self, X, X_star = np.array([])):
        """Compute Covariance Matrix for the Periodic kernel K(X,X) or K(X, X_star)

        Args:
            X (n x p numpy matrix) : Data on which to compute
            X_star (optional, n x p numpy matrix) : Second dataset
        """

        if X_star.shape[0]:
            K_ = distance_matrix(X, X_star)
        else:
            K_ = distance_matrix(X, X)

        part1 = 2 ** (1 - self.mu) / gamma(self.mu)
        part2 = (np.sqrt(2 * self.mu) * K_ / self.lengthscale) ** self.mu
        part3 = kv(self.mu, np.sqrt(2 * self.mu) * K_ / self.lengthscale)
        return ((self.var)**2) * part1 * part2 * part3

    def set_values(self, values):
        """Compute Updates kernel for Periodic Kernel"""

        kernel = Periodic(values[0], values[1], values[2])
        return kernel

    def get_values(self):
        """Return the parameters of the kernel"""

        return [self.lengthscale, self.period, self.var]

class RQ(Kernel):
    """Instantiate a Rational Quadratic Kernel

    Args:
        l : lengthscale
        alpha : exponent
        var : variance

    """

    def __init__(self, l = 1., alpha = 1., var = 1.):
        super().__init__()
        self.lengthscale = l
        self.alpha = l
        self.var = var
        self.n_params = 3

    def compute_covariance(self, X, X_star = np.array([])):
        """Compute Covariance Matrix for the Periodic kernel K(X,X) or K(X, X_star)

        Args:
            X (n x p numpy matrix) : Data on which to compute
            X_star (optional, n x p numpy matrix) : Second dataset
        """

        if X_star.shape[0]:
            K_ = distance_matrix(X, X_star)
        else:
            K_ = distance_matrix(X, X)

        K_ =  (1 + K_ ** 2 / (2*self.alpha * self.lengthscale**2))**(-self.alpha)

        return (self.var**2)*K_

    def set_values(self, values):
        """Compute Updates kernel for Periodic Kernel"""

        kernel = RQ(values[0], values[1], values[2])
        return kernel

    def get_values(self):
        """Return the parameters of the kernel"""
        return [self.lengthscale, self.alpha, self.var]
