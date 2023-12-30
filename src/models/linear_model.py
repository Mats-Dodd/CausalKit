from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class LinearModel(ABC):
    """
    Abstract class to represent a general Linear Model.
    """

    def __init__(self, df: pd.DataFrame, outcome: str, independent: list, intercept: bool = True):
        self.data = df
        self.outcome = outcome
        self.independent_vars = independent
        self.intercept = intercept
        self.dependent_data = self.data[self.outcome].values
        self.obs = len(self.data)

        self._initialize_attributes()
        self._set_attributes()
        self._fit()

    def _initialize_attributes(self):
        """
        Initialize basic attributes of the model.
        """
        self.independent_data = None
        self.n_regs = None
        self.degrees_of_freedom = None
        self.k = None
        self.coefficients = None

    def _set_attributes(self):
        """
        Set additional attributes based on the independent variables.
        """
        self._parse_independent_vars()
        self.n_regs = len(self.independent_vars)
        self.degrees_of_freedom = self.obs - self.n_regs - (1 if self.intercept else 0)
        self.k = self.n_regs + (1 if self.intercept else 0)

    def _parse_independent_vars(self):
        """
        Parse the independent variables to the correct format.
        """
        for variable in self.independent_vars:

            if '*' in self.independent_vars:
                self._set_all_operator()
            else:
                self.independent_data = self.data[self.independent_vars].values

    def _set_all_operator(self):
        """
        Use all columns except for the outcome as independent variables.
        """
        self.independent_vars = [col for col in self.data.columns if col != self.outcome]
        self.independent_data = self.data[self.independent_vars].values

    def _add_intercept(self, x):
        """
        Add an intercept to the independent variable data if required.
        """
        if self.intercept:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            return np.c_[np.ones((x.shape[0], 1)), x]
        return x

    @abstractmethod
    def _fit(self):
        """
        Abstract method to fit the model. Must be implemented in subclasses.
        """
        pass

    def predict(self, x_new):
        """
        Make predictions using the fitted model.
        """
        if not isinstance(x_new, np.ndarray):
            x_new = np.array([x_new])
        if self.intercept:
            x_new = self._add_intercept(x_new)
        return x_new @ self.coefficients

    def fitted_values(self):
        """
        Calculate the fitted values of the model.
        """
        x = self.independent_data
        if self.intercept:
            x = self._add_intercept(x)
        return x @ self.coefficients

    def residuals(self):
        """
        Compute the residuals of the model.
        """
        return self.data[self.outcome] - self.fitted_values()
