import numpy as np
import pandas as pd
from IPython.display import display, HTML

from src.models.linreg import LinReg


class IV(LinReg):

    """
    Class to represent an instrumental variable Regression model.
    Inherits from the LinReg class and adds functionality to handle instrumental variables.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 outcome: str,
                 independent: list,
                 instruments: list,
                 intercept=True,
                 standard_error_type='non-robust'):
        """

        Initialize the FixedEffectsModel.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        outcome (str): The name of the outcome column.
        independent (list): A list of names of independent variables.
        fixed (list): A list of column names for which to include fixed effects.
        intercept (bool): Whether to include an intercept in the model.
        standard_error_type (str): The type of standard error calculation.
        """
        self.instruments = instruments
        self.model_type = 'Instrumental Variables'
        self.first_stage = None
        self.second_stage = None
        super().__init__(df, outcome, independent, intercept, standard_error_type)

    def _first_stage(self):
        """
        Calculate the first stage regression.
        """
        self.first_stage = LinReg(self.df,
                                  self.instruments,
                                  self.independent,
                                  self.intercept,self.standard_error_type)


    def _second_stage(self):
        pass

    def _fit(self):
        pass
