import pandas as pd
from IPython.display import display, HTML

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
        self.standard_error_type = standard_error_type
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
                                        independent=independent,
                                        standard_error_type=self.standard_error_type)

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
                                         independent,
                                         standard_error_type=self.standard_error_type)

    def _fit(self):
        """
        Fit the Instrumental Variables model.
        """
        self._first_stage()
        self._second_stage()

        self.summary_data_coefficients = self.second_stage_model.summary_data_coefficients
        self.summary_data_model = self.second_stage_model.summary_data_model
        self.table = self.second_stage_model.table

    def predict(self, x_new):
        """
        Make predictions using the fitted IV model.
        """
        # Prediction would typically use second-stage coefficients
        return self.second_stage_model.predict(x_new)

    def summary(self, **kwargs):
        # Define the first and second stage equations as strings
        first_stage_equation = f"{self.independent_var[0]} ~ {' + '.join(self.instruments + self.controls)}"
        second_stage_equation = f"{self.outcome} ~ {' + '.join(['predicted_' + self.independent_var[0]] + self.controls)}"

        # Start the HTML string with the title and the first and second stage equations as regular text
        html = "<p style='text-align:center; font-size:20px;'><strong>Instrumental Variables Regression Results</strong></p>"
        html += f"<p style='text-align:center;'>First Stage Equation: {first_stage_equation}</p>"
        html += f"<p style='text-align:center;'>Second Stage Equation: {second_stage_equation}</p>"
        html += "<pre style='text-align:center; font-family:monospace;'>"

        # Retrieve the summary data model from the second stage model
        summary_data_model = self.summary_data_model
        summary_data_coefficients = self.summary_data_coefficients

        # Display the model summary as before
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
