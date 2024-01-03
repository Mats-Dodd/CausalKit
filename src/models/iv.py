import pandas as pd

from src.models.linreg import LinReg


class IV(LinReg):
    """
    Class to represent an Instrumental Variable Regression model.
    Inherits from the LinReg class and adds functionality to handle instrumental variables.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 outcome: str,
                 independent: list,
                 controls: list,
                 instruments: list,
                 intercept=True,
                 standard_error_type='non-robust'):
        """
        Initialize the Instrumental Variables model.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        outcome (str): The name of the outcome column.
        independent (list): A list of names of independent variables.
        instruments (list): A list of names of instrumental variables.
        intercept (bool): Whether to include an intercept in the model.
        standard_error_type (str): The type of standard error calculation.
        """
        self.independent_var = independent
        self.controls = controls
        self.instruments = instruments
        self.model_type = 'Instrumental Variables'
        self.first_stage_model = None
        self.second_stage_model = None
        super().__init__(df, outcome, independent, intercept, standard_error_type)

    def _first_stage(self):
        """
        Calculate the first stage regression.
        """
        df = self.data.copy()
        outcome = str(self.independent_var[0])
        independent = self.instruments + self.controls
        self.first_stage_model = LinReg(df=df,
                                        outcome=outcome,
                                        independent=independent)

    def _second_stage(self):
        """
        Calculate the second stage regression.
        """

        fitted_independent = self.first_stage_model.fitted_values()
        second_stage_df = self.data.copy()
        second_stage_df['independent_hat'] = fitted_independent

        independent = ['independent_hat'] + self.controls
        self.second_stage_model = LinReg(second_stage_df,
                                         self.outcome,
                                         independent)

    def _fit(self):
        """
        Fit the Instrumental Variables model.
        """
        self._first_stage()
        self._second_stage()

    def predict(self, x_new):
        """
        Make predictions using the fitted IV model.
        """
        # Prediction would typically use second-stage coefficients
        return self.second_stage_model.predict(x_new)
