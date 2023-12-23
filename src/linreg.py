import numpy as np
import pandas as pd


class LinReg:

    """ Class to represent a Linear Regression model"""

    def __init__(self,
                 df: pd.DataFrame,
                 outcome: str,
                 independent: [str],
                 intercept: bool = True,
                 standard_error_type: str = ['non-robust', 'robust', 'clustered'][0]):
        self.data = df
        self.outcome = outcome
        self.independent = independent
        self.intercept = intercept
        self.standard_error_type = standard_error_type
        self.coefficients = []

        self._fit()

    def _add_intercept(self):

        if self.intercept:
            independent_cols = self.data[self.independent]
            self.independent = np.c_[np.ones((len(independent_cols), 1)), independent_cols]

    def _fit_coefficients(self):
        dep = self.data[self.outcome]
        self.coefficients = np.linalg.inv(self.independent.T @ self.independent) @ self.independent.T @ dep

    def _fit(self):
        self._add_intercept()
        self._fit_coefficients()

    def predict(self, new_independent):

        if self.intercept:
            if type(new_independent) is int:
                data = np.c_[np.ones((1, 1)), new_independent]
            else:
                data = np.c_[np.ones((len(new_independent), 1)), new_independent]
        else:
            data = new_independent

        return data @ self.coefficients

    def fitted_values(self):

        return self.independent @ self.coefficients

    def residuals(self):

        return self.data[self.outcome] - self.fitted_values()

    def standard_errors(self):

        if self.standard_error_type == 'non-robust':

            return np.sqrt(np.diag(np.linalg.inv(self.independent.T @ self.independent)))
        elif self.standard_error_type == 'robust':

            return np.sqrt(np.diag(np.linalg.inv(self.independent.T @ self.independent) @
                                    self.independent.T @ np.diag(self.residuals() ** 2) @ self.independent))
        elif self.standard_error_type == 'clustered':
            pass
