"""Module for Gaussian Process Model"""
import numpy as np
import scipy
from scipy.optimize import minimize
from utils import extend_list
from tqdm import tqdm

class GaussianProcess():
    """Implementation of the Gaussian Process Model for Regression"""

    def __init__(self, kernel, noise = 0.):
        """
        Parameter of the Gaussian Process.

        Args:
            mean : considered 0
            kernel : covariance function, class Kernel
            noise : observation noise
        """

        self.kernel = kernel
        self.noise = noise

    def fit(self, data, observation):
        """
        Compute the Covariance Function of the Data.

        Args:
            data (n x p matrix) : Data Points
            observation (n x 1 matrix) : Vector of observation
        """

        if len(data.shape) == 1:
            data = data.reshape(-1,1)

        self.y = observation
        self.X = data

        K_noise = self.kernel.compute_covariance(data) + (self.noise**2)*np.eye(data.shape[0])

        self.L = np.linalg.cholesky(K_noise)


        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))


    def predict(self, X_star):
        """
        Predict the posterior mean and covariance function.

        Args:
            X_star (n x p array) : Test datapoints
        Returns:
            f_star_mean, f_star_cov : Mean / Covariance Posterior

        TO DO : Find a way to not compute the inverse of L
        """
        if len(X_star.shape) == 1:
            X_star = X_star.reshape(-1,1)

        self.L_inv = np.linalg.pinv(self.L)

        K_Xstar_X = self.kernel.compute_covariance(X_star, self.X)
        f_star_mean = K_Xstar_X.dot(self.alpha)

        self.v = scipy.linalg.solve_triangular(self.L, K_Xstar_X.T, lower = True)

        f_star_cov = self.kernel.compute_covariance(X_star) - self.v.T.dot(self.v)

        return f_star_mean, f_star_cov

    def log_marginal_likelihood(self):
        """Compute the opposite of the log marginal likelihood for optimization"""

        return 0.5* self.y.dot(self.alpha) + 0.5 * self.y.shape[0] * np.log(2*np.pi) + np.sum(np.log(np.diag(self.L)))

    def log_test_likelihood(self, t_true_test, y_true_test, f_star_mean, f_star_cov):
        """Compute the opposite of the log test likelihood for test performance"""

        if len(t_true_test.shape) == 1:
            k = 1
        else:
            k = t_true_test.shape[1]
        f_star_cov += (self.noise**2)*np.eye(f_star_cov.shape[0])

        m_log_likelihood = 0.5 * k * np.log(2*np.pi) + 0.5 * np.linalg.det(f_star_cov) + 0.5 * (y_true_test - f_star_mean).T.dot(np.linalg.pinv(f_star_cov).dot(y_true_test - f_star_mean))
        return m_log_likelihood

    def optimize(self, initial_values = [], bounds = None):
        """Optimize the model hyperparameters with the given kernel form
        Args:
            - initial_values (Optional, list or array): initial values for optimization,
            if not specified : take the kernel Hyperparameters
            - bounds (Optional, List of couples) : bounds for each parameters,
            if a None values are given, then the corresponding parameters
            are fixed during optimization
        """

        def obj(values):
            kernel = self.kernel.set_values(values[:(-1)])
            model = GaussianProcess(kernel, noise = values[-1])

            try:
                model.fit(self.X, self.y)
                assessment = model.log_marginal_likelihood()
            except Exception as e:
                print("The following parameters failed:", values)
                print(e)
                assessment = np.inf
            return assessment

        if len(initial_values) == 0:
            initial_values = self.kernel.get_values()
            initial_values.append(self.noise)


        initial_vector = initial_values
        extend = lambda x : x

        if bounds is not None:
            trainable_parameters_index = np.array([i for i, el in enumerate(bounds) if el is not None])
            if len(trainable_parameters_index) != len(initial_values):
                print("Fixing parameters...")
                extend = extend_list(trainable_parameters_index, np.array(initial_values))
                initial_vector = np.array(initial_values)[trainable_parameters_index]
                bounds = [bounds[i] for i in trainable_parameters_index]

        objective = lambda x : obj(extend(x))

        result = minimize(objective, initial_vector, method='L-BFGS-B', bounds = bounds)
        print("-----Result Optimization-----")
        print("Convergence:", result["success"])
        print("Value Log Marginal Likelihood:", -result["fun"])

        optimal_values = extend(result["x"])
        print("Value Parameters:", optimal_values)

        self.kernel = self.kernel.set_values(optimal_values[:(-1)])
        self.noise = optimal_values[-1]
        self.fit(self.X, self.y)

        return optimal_values


    def optimize_log(self, initial_values = [], bounds = None):
        """Optimize the model hyperparameters with the given kernel form in log space
        Args:
                initial_values (Optional, list or array): initial values for optimization,
            if not specified : take the kernel Hyperparameters

                bounds (Optional, List of couples) : bounds for each parameters,
            if a None values are given, then the corresponding parameters
            are fixed during optimization
        """

        def obj(values):
            kernel = self.kernel.set_values(values[:(-1)])
            model = GaussianProcess(kernel, noise = values[-1])

            try:
                model.fit(self.X, self.y)
                assessment = model.log_marginal_likelihood()
            except Exception as e:
                print("The following parameters failed:", values)
                print(e)
                assessment = np.inf
            return assessment

        if len(initial_values) == 0:
            initial_values = self.kernel.get_values()
            initial_values.append(self.noise)


        initial_vector = initial_values
        extend = lambda x : x

        if bounds is not None:
            bounds = [(np.log(el[0]), np.log(el[1])) if el is not None else el for el in bounds ]
            trainable_parameters_index = np.array([i for i, el in enumerate(bounds) if el is not None])
            if len(trainable_parameters_index) != len(initial_values):
                print("Fixing parameters...")
                extend = extend_list(trainable_parameters_index, np.array(initial_values))
                initial_vector = np.array(initial_values)[trainable_parameters_index]
                bounds = [bounds[i] for i in trainable_parameters_index]

        objective = lambda x : obj(extend(np.exp(x)))

        result = minimize(objective, np.log(initial_vector), method='L-BFGS-B', bounds = bounds)
        print("-----Result Optimization-----")
        print("Convergence:", result["success"])
        print("Value Log Marginal Likelihood:", -result["fun"])

        optimal_values = extend(np.exp(result["x"]))
        print("Value Parameters:", optimal_values)

        self.kernel = self.kernel.set_values(optimal_values[:(-1)])
        self.noise = optimal_values[-1]
        self.fit(self.X, self.y)

        return optimal_values


    def sequential_prediction(self, t_train, observation):
        """Prediction of the observation in a sequential manner

        Args:
            data (n x 1 array) : Data Points
            observation (n x 1 array) : Vector of observation

        TO DO : Optimize complexity
        """
        n = t_train.shape[0]
        mean_predictive_sequence = []
        var_predictive_sequence = []
        t_test_final = []
        for i in tqdm(range(1, n-1)):
            data_seq = t_train[:i]
            observation_seq = observation[:i]

            # Take linspace of the two data points
            t_test_seq = np.linspace(t_train[i], t_train[i+1,], int(100*(t_train[i+1] - t_train[i])))

            self.fit(data_seq, observation_seq)
            result_seq = self.predict(t_test_seq)
            mean_predictive_sequence += list(result_seq[0])
            var_predictive_sequence += list(np.diag(result_seq[1]))
            t_test_final += list(t_test_seq)
        return np.array(t_test_final), np.array(mean_predictive_sequence), np.diag(var_predictive_sequence)
