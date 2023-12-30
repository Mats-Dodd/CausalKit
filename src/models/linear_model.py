from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class LinearModel(ABC):

    """ Abstract class to represent a general Linear Model """

    def __init__(self, df: pd.DataFrame, outcome: str, independent: list, intercept: bool = True):
        self.data = df
        self.outcome = outcome
        self.independent_vars = independent
        self.intercept = intercept
        self.independent_data = self.data[self.independent_vars].values
        self.dependent_data = self.data[self.outcome].values
        self.n_regs = len(independent)
        self.obs = len(self.data)
        self.degrees_of_freedom = self.obs - self.n_regs - (1 if self.intercept else 0)
        self.k = self.n_regs + (1 if self.intercept else 0)
        self.coefficients = None

        self._fit()

    def _add_intercept(self, x):
        if self.intercept:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            return np.c_[np.ones((x.shape[0], 1)), x]
        return x

    @abstractmethod
    def _fit(self):
        pass

    def predict(self, x_new):
        if not isinstance(x_new, np.ndarray):
            x_new = np.array([x_new])
        if self.intercept:
            x_new = self._add_intercept(x_new)

        return x_new @ self.coefficients

    def fitted_values(self):
        x = self.independent_data
        if self.intercept:
            x = self._add_intercept(x)

        return x @ self.coefficients

    def residuals(self):
        return self.data[self.outcome] - self.fitted_values()