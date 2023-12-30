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

        '*': all columns except for the outcome as independent variables,
        '~': all columns except those that are equal to outcome or have a prefix '~' in self.independent_vars,
        ':': interaction between those specific variables,
        '*': interaction between those two + their normal values,
        '^': to the power of the number after the '^'
        """
        if any(char in var for var in self.independent_vars for char in ['.', ':', '*', '!', '^']):

            contains_caret = any('^' in var for var in self.independent_vars)
            contains_colon = any(':' in var for var in self.independent_vars)
            contains_no_star_or_colon = all('*' not in var and ':' not in var for var in self.independent_vars)
            contains_no_caret = all('^' not in var for var in self.independent_vars)
            contains_colon_or_star = any('*' in var or ':' in var for var in self.independent_vars)
            if any('.' in var for var in self.independent_vars):
                self._set_all_operator()

            elif any('!' in var for var in self.independent_vars):
                self._set_not_operator()

            elif contains_caret and contains_no_star_or_colon:
                self._set_polynomial_operator()

            elif contains_no_caret and contains_colon_or_star:

                if contains_colon:
                    self._set_basic_interaction_operator()
                else:
                    self._set_advanced_interaction_operator()

        else:

            self.independent_data = self.data[self.independent_vars].values

    def _set_all_operator(self):
        """
        Use all columns except for the outcome as independent variables.
        """
        self.independent_vars = [col for col in self.data.columns if col != self.outcome]
        self.independent_data = self.data[self.independent_vars].values

    def _set_not_operator(self):
        """
        Use all columns from self.data.columns as independent variables,
        except those that are equal to outcome or have a prefix '~' in self.independent_vars.
        """
        excluded_vars = {var.lstrip('!') for var in self.independent_vars if var.startswith('!')}

        for var in excluded_vars:
            if var not in self.data.columns:
                raise ValueError(f'Oops, {var} is not a column in your data. Check if youve made a typo.')

        if len(excluded_vars) == len(self.data.columns) - 1:
            raise ValueError('Oops, you cant exclude all columns from your data.')

        self.independent_vars = [col for col in self.data.columns if col != self.outcome and col not in excluded_vars]
        self.independent_data = self.data[self.independent_vars].values

    def _set_polynomial_operator(self):
        """
        Set the independent variables to the power of the number after the '^' and also include all lower powers.
        """
        new_vars = set()
        for var in self.independent_vars:
            if '^' in var:
                base_var, exponent_str = var.split('^')
                try:
                    max_exponent = int(exponent_str)
                except ValueError:
                    raise ValueError(f"Invalid exponent '{exponent_str}' in variable '{var}'. Check for a typo.")

                if base_var not in self.data.columns:
                    raise ValueError(f"Base variable '{base_var}' not found in DataFrame. Check for a typo in '{var}'.")

                for exponent in range(2, max_exponent + 1):
                    transformed_col_name = f"{base_var}^{exponent}"
                    self.data[transformed_col_name] = self.data[base_var] ** exponent
                    new_vars.add(transformed_col_name)
            else:
                new_vars.add(var)

        self.independent_vars = list(new_vars)
        self.independent_data = self.data[self.independent_vars].values

    def _set_basic_interaction_operator(self):
        """
        Set the interaction between those specific variables.
        """
        new_vars = set()
        for var in self.independent_vars:
            if ':' in var:
                var1, var2 = var.split(':')
                if var1 not in self.data.columns:
                    raise ValueError(f"Variable '{var1}' not found in DataFrame. Check for a typo in '{var}'.")
                if var2 not in self.data.columns:
                    raise ValueError(f"Variable '{var2}' not found in DataFrame. Check for a typo in '{var}'.")

                transformed_col_name = f"{var1}:{var2}"
                self.data[transformed_col_name] = self.data[var1] * self.data[var2]
                new_vars.add(transformed_col_name)
            else:
                new_vars.add(var)

        self.independent_vars = list(new_vars)
        self.independent_data = self.data[self.independent_vars].values

    def _set_advanced_interaction_operator(self):
        """
        Set the interaction between two variables and include their individual terms.
        """
        new_vars = set()
        for var in self.independent_vars:
            if '*' in var:
                var1, var2 = var.split('*')
                # Validate existence of variables in DataFrame
                if var1 not in self.data.columns:
                    raise ValueError(f"Variable '{var1}' not found in DataFrame. Check for a typo in '{var}'.")
                if var2 not in self.data.columns:
                    raise ValueError(f"Variable '{var2}' not found in DataFrame. Check for a typo in '{var}'.")

                # Add individual variables
                new_vars.add(var1)
                new_vars.add(var2)

                # Create and add interaction term
                transformed_col_name = f"{var1}*{var2}"
                self.data[transformed_col_name] = self.data[var1] * self.data[var2]
                new_vars.add(transformed_col_name)
            else:
                new_vars.add(var)

        self.independent_vars = list(new_vars)
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
