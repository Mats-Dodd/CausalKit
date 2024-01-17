import numpy as np
import pandas as pd
import scipy.stats as stats
from IPython.display import display, HTML

from src.models.linear_model import LinearModel


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _compute_gradient(x, y, predictions):
    """
    Compute the gradient of the log-likelihood.
    :param x:
    :param y:
    :param predictions:
    :return: gradient of the log-likelihood
    """
    return x.T @ (y - predictions)


def _compute_hessian(x, predictions):
    """
    Compute the Hessian of the log-likelihood.
    :param x:
    :param predictions:
    :return: hessian
    """

    w = np.diag(predictions * (1 - predictions))

    return x.T @ w @ x


def _compute_log_likelihood(y, predictions):
    """
    Compute the log-likelihood.
    :param y:
    :param predictions:
    :return: log-likelihood
    """
    return np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))


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
        self._set_wald_statistics()
        self._set_conf_int()

        self._set_degrees_of_freedom()
        self._set_log_likelihood()
        self._set_log_likelihood_null()
        self._set_pseudo_r_squared()
        self._set_llr_p_value()
        self._set_summary_data()
        self._set_table()

    def predict(self, x_new):
        """
        Predict the probability of a positive outcome for a new observation.
        """
        x_new = np.array(x_new, dtype=float).reshape(1, -1)

        if self.intercept:
            x_new = self._add_intercept(x_new)

        return _sigmoid(x_new @ self.coefficients)

    def predict_class(self, x_new, threshold=0.5):
        """
        Predict the class of a new observation.
        """
        prediction = self.predict(x_new) >= threshold
        return prediction.astype(int)

    def fitted_values(self):
        """
        Return the fitted values of the model.
        """
        x = self.independent_data
        if self.intercept:
            x = self._add_intercept(x)
        return _sigmoid(x @ self.coefficients)

    def residuals(self, residual_type='pearson'):
        """
        Return the pearson residuals of the model.
        """
        fitted = self.fitted_values()
        return self.dependent_data - fitted / np.sqrt(fitted * (1 - fitted))

    def marginal_effects(self, at='mean', method='dydx'):
        """
        Compute marginal effects for logistic regression.

        :param at: Specify whether to evaluate at 'mean' or 'median' of independent variables.
        :param method: Specify the method to use, 'dydx' for marginal effect, 'eyex' for elasticity.
        :return: DataFrame with marginal effects for each independent variable.
        """
        if at not in ['mean', 'median']:
            raise ValueError("The 'at' parameter must be 'mean' or 'median'")

        if method not in ['dydx', 'eyex']:
            raise ValueError("The 'method' parameter must be 'dydx' or 'eyex'")

        x = self.independent_data
        if self.intercept:
            x = self._add_intercept(x)

        # Getting the mean or median values of the independent variables
        x_at_values = np.mean(x, axis=0) if at == 'mean' else np.median(x, axis=0)

        # Predict probabilities at specified values
        prob_at_values = _sigmoid(x_at_values @ self.coefficients)

        # Calculate marginal effects
        marginal_effects = {}
        for idx, var in enumerate(['Intercept'] + self.independent_vars):
            if method == 'dydx':

                effect = prob_at_values * (1 - prob_at_values) * self.coefficients[idx]
            elif method == 'eyex':
                # Elasticity (percentage change in probability for 1% change in variable)
                """TODO"""
                pass
            marginal_effects[var] = effect

        return pd.DataFrame(marginal_effects, index=[0])

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
        """
        Compute robust standard errors using the HC0 (Huber-White) estimator.
        """
        predictions = _sigmoid(x @ self.coefficients)

        residuals = y - predictions

        xtx_inv = np.linalg.pinv(x.T @ x)

        s = np.diag(residuals ** 2)
        robust_variance = xtx_inv @ (x.T @ s @ x) @ xtx_inv
        self.standard_errors = np.sqrt(np.diag(robust_variance))

    def _fit_standard_errors(self):
        x = self.independent_data
        if self.intercept:
            x = self._add_intercept(x)
        y = self.dependent_data

        if self.standard_error_type == 'non-robust':

            self._compute_normal_se(x)

        elif self.standard_error_type == 'hc0':
            self._compute_robust_se(x, y)

        else:
            raise ValueError('Invalid standard error type.')

    def _set_wald_statistics(self):
        """
        Compute the Wald statistics (Z-scores) and their corresponding p-values.
        """
        self.wald_stats = self.coefficients / self.standard_errors
        self.p_values = stats.norm.sf(np.abs(self.wald_stats)) * 2

    def _set_conf_int(self,  alpha=0.05):
        """
        Compute the 95% confidence intervals for the model coefficients.
        """
        z = stats.norm.ppf(1 - alpha / 2)
        lower_bounds = self.coefficients - z * self.standard_errors
        upper_bounds = self.coefficients + z * self.standard_errors

        self.conf_int = [[lower, upper] for lower, upper in zip(lower_bounds, upper_bounds)]

    def _set_degrees_of_freedom(self):
        self.df_residuals = self.obs - len(self.coefficients)
        self.df_model = len(self.coefficients) - 1

    def _set_log_likelihood(self):
        """
        Compute the log-likelihood of the model.
        """
        x = self.independent_data
        if self.intercept:
            x = self._add_intercept(x)
        y = self.dependent_data
        predictions = _sigmoid(x @ self.coefficients)
        self.log_likelihood = _compute_log_likelihood(y, predictions)

    def _set_log_likelihood_null(self):
        y = self.dependent_data
        null_intercept = np.mean(y)
        null_predictions = np.repeat(null_intercept, len(y))
        self.log_likelihood_null = _compute_log_likelihood(y, null_predictions)

    def _set_pseudo_r_squared(self):
        self.pseudo_r_squared = 1 - (self.log_likelihood / self.log_likelihood_null)

    def _set_llr_p_value(self):
        """
        Compute the Likelihood Ratio Test (LLR) p-value.
        """

        log_likelihood_model = self.log_likelihood
        log_likelihood_null = self.log_likelihood_null

        llr_statistic = 2 * (log_likelihood_model - log_likelihood_null)
        self.llr_p_value = stats.chi2.sf(llr_statistic, df=self.df_model)

    def _set_summary_data(self):
        self.summary_data_coefficients = {
            'Variable': ['Intercept'] + self.independent_vars,
            'Coefficient': [round(num, 3) for num in self.coefficients],
            'Std-Error': [round(num, 3) for num in self.standard_errors],
            'Wald Z-Statistic': [round(num, 3) for num in self.wald_stats],
            'P>|t|': [round(num, 3) for num in self.p_values],
            'Conf. Interval': [[round(num, 3) for num in sublist] for sublist in self.conf_int]
        }

        self.summary_data_model = {
            'Dep. Variable': self.outcome,
            'Observations': self.obs,
            'Model': 'Logit',
            'Method': 'MLE',
            'Df Residuals': self.df_residuals,
            'Df Model': self.df_model,
            'Pseudo R-squared': round(self.pseudo_r_squared, 3),
            'Log-Likelihood': round(self.log_likelihood, 3),
            'LL-Null': round(self.log_likelihood_null, 3),
            'LLR p-value': round(self.llr_p_value, 3)

        }

    def _set_table(self):
        data = self.summary_data_coefficients.copy()
        data['Lower Bound'] = [ci[0] for ci in data['Conf. Interval']]
        data['Upper Bound'] = [ci[1] for ci in data['Conf. Interval']]
        del data['Conf. Interval']

        self.table = pd.DataFrame(data)

    def summary(self):
        """
        Generate a summary of the logistic regression results.
        """
        summary_data_coefficients = self.summary_data_coefficients
        summary_data_model = self.summary_data_model

        html = "<h1 style='text-align:center;'>Logistic Regression Results</h1><pre style='text-align:center; font-family:monospace;'>"

        first_half_model = list(summary_data_model.items())[:len(summary_data_model)//2]
        second_half_model = list(summary_data_model.items())[len(summary_data_model)//2:]

        max_key_len = max(len(key) for key, _ in summary_data_model.items()) + 2

        for (key1, value1), (key2, value2) in zip(first_half_model, second_half_model):
            key1_formatted = f"{key1 + ': ':<{max_key_len}}"
            key2_formatted = f"{key2 + ': ':<{max_key_len}}"
            html += f"{key1_formatted}{str(value1).rjust(10)}    {key2_formatted}{str(value2).rjust(10)}\n"

        html += "\n"

        column_widths = {key: max(max([len(str(x)) for x in summary_data_coefficients[key]]), len(key)) for key in summary_data_coefficients.keys()}
        column_widths['Conf. Interval'] = max(max([len(f"{x[0]} - {x[1]}") for x in summary_data_coefficients['Conf. Interval']]), len('Conf. Interval'))

        header_line = ' '.join(key.center(column_widths[key]) for key in summary_data_coefficients.keys())
        html += header_line.center(len(header_line) + max_key_len) + "\n"

        separator = '-'.join('-' * column_widths[key] for key in summary_data_coefficients.keys())
        html += separator.center(len(header_line) + max_key_len) + "\n"

        for i in range(len(summary_data_coefficients['Variable'])):
            row = []
            for key in summary_data_coefficients.keys():
                if key == 'Conf. Interval':
                    ci_text = f"{summary_data_coefficients[key][i][0]} - {summary_data_coefficients[key][i][1]}"
                    row.append(ci_text.center(column_widths[key]))
                else:
                    row.append(str(summary_data_coefficients[key][i]).center(column_widths[key]))
            html += ' '.join(row).center(len(header_line) + max_key_len) + "\n"

        html += "</pre>"

        return display(HTML(html))
