import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import expit

from src.models.linear_model import LinearModel


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _compute_gradient(x, y, predictions):
    # Gradient of log-likelihood
    return x.T @ (y - predictions)


def _compute_hessian(x, predictions):
    # Diagonal matrix of predictions
    w = np.diag(predictions * (1 - predictions))
    # Hessian matrix of log-likelihood
    return x.T @ w @ x


class LogReg(LinearModel):
    """
    Class to represent a Logistic Regression model.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 outcome: str,
                 independent: list,
                 intercept: bool = True,
                 standard_error_type: str = 'non-robust'):

        self.standard_error_type = standard_error_type

        super().__init__(df,
                         outcome,
                         independent,
                         intercept)

    def _initialize_metrics(self):
        """ Initialize all metrics to None. """
        self.log_likelihood = self.aic = self.bic = None
        self.standard_errors = self.wald_stats = self.p_values = self.conf_int = None

    def _fit(self):
        self._initialize_metrics()
        self._fit_coefficients()
        self._fit_standard_errors()
        self._compute_wald_statistics()
        self._compute_conf_int()

    def _fit_coefficients(self, iterations=10, tol=1e-6):
        x = self.independent_data
        if self.intercept:
            x = self._add_intercept(x)
        y = self.dependent_data

        self.coefficients = np.zeros(x.shape[1])
        count = 0
        for _ in range(iterations):
            pred = x @ self.coefficients
            predictions = _sigmoid(pred)
            gradient = _compute_gradient(x, y, predictions)
            hessian = _compute_hessian(x, predictions)

            update = np.linalg.inv(hessian) @ gradient
            self.coefficients += update
            count += 1
            if np.linalg.norm(update) < tol:
                break
        print(f'Converged in {count} iterations.')

    def _compute_normal_se(self, x):
        hessian = _compute_hessian(x, _sigmoid(x @ self.coefficients))
        hessian_inv = np.linalg.inv(hessian)
        self.standard_errors = np.sqrt(np.diag(hessian_inv))

    def _compute_robust_se(self, x, y):
        """TODO: Implement robust standard errors."""
        predictions = _sigmoid(x @ self.coefficients)
        residuals = y - predictions

        s = np.diag(residuals ** 2)

        gradient_outer = np.zeros((x.shape[1], x.shape[1]))
        for i in range(len(residuals)):
            xi = x[i, :].reshape(-1, 1)
            gradient_outer += xi @ xi.T * residuals[i] ** 2

        hessian = _compute_hessian(x, predictions)
        hessian_inv = np.linalg.inv(hessian)
        robust_covariance = hessian_inv @ gradient_outer @ hessian_inv
        self.standard_errors = np.sqrt(np.diag(robust_covariance))

    def _fit_standard_errors(self):
        print('Fitting standard errors in logreg...')
        x = self.independent_data
        if self.intercept:
            x = self._add_intercept(x)
        y = self.dependent_data

        if self.standard_error_type == 'non-robust':

            self._compute_normal_se(x)
            print(self.standard_errors)

        elif self.standard_error_type == 'robust':
            self._compute_robust_se(x, y)

        else:
            raise ValueError('Invalid standard error type.')

    def _compute_wald_statistics(self):
        """
        Compute the Wald statistics (Z-scores) and their corresponding p-values.
        """
        self.wald_stats = self.coefficients / self.standard_errors
        self.p_values = stats.norm.sf(np.abs(self.wald_stats)) * 2

    def _compute_conf_int(self,  alpha=0.05):
        """
        Compute the 95% confidence intervals for the model coefficients.
        """
        z = stats.norm.ppf(1 - alpha / 2)  # z-score for 95% CI
        lower_bounds = self.coefficients - z * self.standard_errors
        upper_bounds = self.coefficients + z * self.standard_errors

        self.conf_int = [[lower, upper] for lower, upper in zip(lower_bounds, upper_bounds)]









