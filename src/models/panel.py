import numpy as np
import pandas as pd
import scipy.stats as stats
from IPython.display import display, HTML

from src.models.linreg import LinReg


class FixedEffects(LinReg):
    """
    Class to represent a Fixed Effects Linear Regression model.
    Inherits from the LinReg class and adds functionality to handle fixed effects.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 outcome: str,
                 independent: list,
                 fixed: list,
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
        self.fixed_vars = fixed
        self.model_type = 'Fixed Effects'
        self.dummy_cols = []
        super().__init__(df, outcome, independent, intercept, standard_error_type)

    def _add_fixed_effects(self):
        """
        Add dummy variables for each category in each fixed effect variable.
        """
        for level in self.fixed_vars:
            if level not in self.data.columns:
                raise ValueError(f"{level} is not a column in the data provided.")

            dummies = pd.get_dummies(self.data[level].astype('string'), prefix=f'dummy_{level}', drop_first=True).astype(int)
            self.dummy_cols.extend(dummies.columns)
            self.data = pd.concat([self.data, dummies], axis=1)
        self.independent_vars.extend(self.dummy_cols)
        self.independent_data = self.data[self.independent_vars]

    def _fit(self):
        """
        Fit the model. Overriding to include fixed effects.
        """
        if len(self.fixed_vars) < 1:
            raise ValueError(f"You have not specified any levels to fix by.  Look at adding a 'fixed' parameter.")

        self._add_fixed_effects()
        super()._fit()

    def _fit_standard_errors(self):
        x = self.independent_data
        if self.intercept:
            x = self._add_intercept(x)

        error_variance = self.rss / self.degrees_of_freedom
        xtx_inv = np.linalg.pinv(x.T @ x)

        if self.standard_error_type == 'non-robust':
            self.standard_errors = np.sqrt(error_variance * np.diag(xtx_inv))

        elif self.standard_error_type == 'robust':
            residuals = self.residuals()
            s = np.diag(residuals ** 2)
            robust_variance = xtx_inv @ (x.T @ s @ x) @ xtx_inv
            self.standard_errors = np.sqrt(np.diag(robust_variance))

        if self.standard_error_type == 'clustered':
            if self.fixed_vars is None:
                raise ValueError("Cluster variable not specified for clustered standard errors.")

            residuals = self.residuals()
            sum_sq_grouped_errors = np.zeros((x.shape[1], x.shape[1]))

            # Group by the cluster variables and iterate over each group
            grouped_data = self.data.groupby(self.fixed_vars)
            for _, group in grouped_data:
                group_idx = group.index
                xi = x[group_idx]
                ri = residuals[group_idx].values.reshape(-1, 1)  # Corrected line
                sum_sq_grouped_errors += xi.T @ ri @ ri.T @ xi

            clustered_variance = np.linalg.pinv(x.T @ x) @ sum_sq_grouped_errors @ np.linalg.pinv(x.T @ x)
            self.standard_errors = np.sqrt(np.diag(clustered_variance))

    def _set_summary_data(self):
        # Call the base method to initialize summary data
        super()._set_summary_data()

        # Filter the data for non-dummy variables
        non_dummy_data = {
            key: [] for key in self.summary_data_coefficients
        }
        dummy_data = {
            key: [] for key in self.summary_data_coefficients
        }

        for i, var in enumerate(self.summary_data_coefficients['Variable']):
            if var not in self.dummy_cols:
                for key in non_dummy_data:
                    non_dummy_data[key].append(self.summary_data_coefficients[key][i])
            else:
                for key in dummy_data:
                    dummy_data[key].append(self.summary_data_coefficients[key][i])

        # Update the class attributes
        self.summary_data_coefficients = non_dummy_data
        self.summary_data_dummies = dummy_data

    def summary(self, content_type='dynamic', show_fixed=False):

        summary_data_coefficients = self.summary_data_coefficients

        summary_data_model = self.summary_data_model

        header_tooltips = {
            'Variable': '''
                This is the independent variable in the regression model, which is the factor being manipulated
                or changed to observe its effect on the dependent variable.
                ''',

            'Coefficient': '''
                This represents the coefficient estimate of the variable, indicating how much the dependent variable
                is expected to change when the independent variable changes by one unit.
                ''',

            'Std-Error': '''
                This stands for the standard error of the coefficient estimate, which measures the average distance that
                the estimated coefficients are from the actual average value of the coefficients.
                ''',

            'T-Statistic': '''
                The T- Statistic helps determining whether there
                is a significant relationship between the independent and dependent variables by comparing the estimated
                coefficient to its standard error.
                ''',

            'P>|t|': '''
                This denotes the p-value, which indicates the probability of observing the data, or something more extreme,
                assuming the null hypothesis is true.
                ''',
            'Conf. Interval': '''
                The 95% CI for the coefficient estimate. If we were to take many samples and build a
                confidence interval from each of them, 95% of these intervals would contain the true coefficient value.
                '''
        }

        model_tooltips = {
            'Dep. Variable': """
                The dependent variable/outcome is the main variable of interest you are trying to
                explain or predict. It is the outcome that changes in response to the independent variables. In a
                regression model, this is what you are modeling as a function of other variables.""",

            'Observations': """
                This refers to the number of data points or individual records used in your regression to estimate the
                model parameters. A higher number of observations can provide more information, but it's also important
                that these observations are representative of the population.""",

            'Standard Error Type': """
                The standard error type indicates the measurement of variability for coefficient estimates. Different
                types (like robust standard errors) can be chosen based on data characteristics. In practice, if
                there's suspicion of heteroskedasticity, using robust standard errors is a common approach""",

            'R-squared': """
                R-squared is a statistical measure that represents the proportion of the variance in the dependent
                variable that is predictable from the independent variables. It ranges from 0 to 1, with higher values
                indicating a better fit of the model. It's a quick way to see how well your model explains the
                variation in the data.""",

            'Adj. R-squared': """
                Adjusted for the number of predictors, it provides a more accurate measure of the model's explanatory
                power. It's lower than R-squared, penalizing excessive use of unhelpful predictors. Ideally,
                 Adj. R-squared should be close to the R-squared value for a well-specified model.""",

            'Log-Likelihood': """
                Reflects the likelihood of observing the given data under the model. Higher values
                indicate better model fit. In model comparisons, a higher log-likelihood generally signifies a better
                model, especially when comparing models with a similar number of parameters.""",

            'AIC': """
                Balances model fit and complexity, penalizing extra parameters. Lower AIC suggests a better model.
                A rule of thumb is that a difference in AIC values of more than 2 indicates a noticeable difference in
                 model quality.""",

            'BIC': """
                Similar to AIC, but with a stricter penalty for model complexity. Lower BIC values indicate a
                better model. BIC is particularly useful in larger datasets and when comparing models with different
                numbers of parameters.""",

            'Adj. AIC': """
            Adjusts AIC for sample size. Like AIC, lower values suggest a better model. This adjustment is especially
            important in smaller samples or when comparing models with vastly different numbers of observations.""",

            'Adj. BIC': """
            Adjusts BIC for sample size, useful in smaller datasets. Lower values indicate a better
            model. Adjusted BIC is crucial for small sample sizes, providing a more accurate comparison of models.""",

            'F-statistic': """
            Tests the overall significance of the model. A higher value indicates a better model. As a rule of thumb,
             an F-statistic greater than 10 is often considered indicative of a strong relationship between the
             dependent and independent variables.""",

            'Prob (F-statistic)': """Indicates the probability of observing the given F-statistic if no independent
            variables affect the dependent variable. Lower values suggest significant model fit. Typically, a
            Prob (F-statistic) less than 0.05 indicates that the model is statistically significant."""

        }

        if content_type == 'dynamic':
            first_half = list(summary_data_model.items())[:len(summary_data_model)//2]
            second_half = list(summary_data_model.items())[len(summary_data_model)//2:]

            html = "<h2 style='text-align: center;'>Regression Results</h2>"

            html += "<div class='model-container'>"

            html += "<div class='model-column'>"
            for key, value in first_half:
                tooltip = model_tooltips.get(key, "")
                html += f"""
                    <div class='model-row'>
                        <div class="model-header">
                            <div class="hover-box">
                                {key}:
                                <div class="hover-content">{tooltip}</div>
                            </div>
                        </div>
                        <div class='model-data'>{value}</div>
                    </div>
                    """
            html += "</div>"

            # Second Column
            html += "<div class='model-column'>"
            for key, value in second_half:
                tooltip = model_tooltips.get(key, "")
                html += f"""
                    <div class='model-row'>
                        <div class="model-header">
                            <div class="hover-box">
                                {key}:
                                <div class="hover-content">{tooltip}</div>
                            </div>
                        </div>
                        <div class='model-data'>{value}</div>
                    </div>
                    """
            html += "</div>"

            html += "</div>"

            html += "</div>"
            html += "<hr style='border-style: dashed;'>"

            html += """
            <style>
                .model-container {
                    display: flex;
                    justify-content: center;
                    margin-bottom: 20px;
                }

                .model-column {
                    display: flex;
                    flex-direction: column;
                    margin-right: 10px;
                }

                .model-row {
                    display: flex;
                    align-items: center;
                    margin-bottom: 5px;
                }

                .model-header {
                    min-width: 150px; /* Adjust this value as needed for alignment */
                }

                .model-header .hover-box {
                    cursor: pointer;
                    position: relative;
                    display: inline-block;
                }

                .model-header .hover-content {
                    display: none;
                    position: absolute;
                    background-color: grey;
                    border: 1px solid black;
                    padding: 15px;
                    width: 350px;
                    z-index: 1;
                    white-space: wrap;
                }

                .model-header:hover .hover-content {
                    display: block;
                }

                .model-data {
                    text-align: left;
                }

                .model-spacer {
                    width: 20px; /* Adjust this width to increase or decrease the space */
                }

                .model-key {
                    flex: 1;
                    text-align: right;
                    margin-right: 10px;
                }

                .model-value {
                    flex: 1;
                    text-align: left;
                }

                .model-header .hover-box {
                    cursor: pointer;
                    position: relative;
                    display: inline-block;
                }

                .model-header .hover-content {
                    display: none;
                    position: absolute;
                    background-color: grey;
                    border: 1px solid black;
                    padding: 15px;
                    width: 350px;
                    z-index: 1;
                    white-space: wrap;
                }

                .model-header:hover .hover-content {
                    display: block;
                    }

            </style>
            """

            html += """
                        <style>
                        .container {
                            display: flex;
                            justify-content: center;
                            align-items: flex-start;
                        }

                        .column {
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            margin-right: 10px;
                        }

                        .header .hover-box {
                            cursor: pointer;
                            position: relative;
                            display: inline-block;
                        }

                        .header .hover-content {
                            display: none;
                            position: absolute;
                            background-color: grey;
                            border: 1px solid black;
                            padding: 15px;
                            width: 350px;
                            z-index: 1;
                            white-space: wrap;
                        }

                        .header:hover .hover-content {
                            display: block;
                        }

                        .data-cell {
                            text-align: center;
                            margin: 2px 0;
                        }

                        .red-light {
                            color: red; /* Light shade for P>|t| < 0.05 */
                        }

                        .red-medium {
                            color: darkred; /* Medium shade for P>|t| < 0.01 */
                        }

                        .red-deep {
                            color: maroon; /* Deep shade for P>|t| < 0.001 */
                        }
                        </style>
                        """

            html += "<div class='container'>"

            for key in summary_data_coefficients.keys():

                html += "<div class='column'>"

                tooltip = header_tooltips.get(key, "")
                html += f"""
                            <div class='header'>
                                <div class="hover-box">
                                    {key}
                                    <div class="hover-content">{tooltip}</div>
                                </div>
                            </div>
                            """

                html += "<hr style='border-style: dashed; width: 100%;'>"

                for value in summary_data_coefficients[key]:
                    if isinstance(value, list):
                        value = f"{value[0]} - {value[1]}"

                    if key == "P>|t|":
                        try:
                            numeric_value = float(value)
                            if numeric_value <= 0.001:
                                cell_class = "data-cell red-deep"
                            elif numeric_value <= 0.01:
                                cell_class = "data-cell red-medium"
                            elif numeric_value <= 0.05:
                                cell_class = "data-cell red-light"
                            else:
                                cell_class = "data-cell"
                        except ValueError:
                            cell_class = "data-cell"
                    else:
                        cell_class = "data-cell"

                    html += f"<div class='{cell_class}'>{value}</div>"

                html += "</div>"

            html += "</div>"

        else:
            html = "<div style='text-align:center; font-family:monospace;'>Regression Results</div>\n"
            html += f"<div style='text-align:center; font-family:monospace;'>Model Type: {self.model_type}</div>\n"
            html += "<pre style='text-align:center; font-family:monospace;'>\n"

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

            if show_fixed:
                #html += "<hr style='border-style: dashed;'>\n"
                html += "<div style='text-align:center;'>Fixed Variables Summary</div>\n"
                html += "<pre style='text-align:center; font-family:monospace;'>\n"

                dummy_data = self.summary_data_dummies
                # Find the maximum length of the content for each column
                column_widths_dummy = {key: max(len(key), max(len(str(val)) for val in dummy_data[key])) + 2 for key in dummy_data.keys()}  # add 2 for padding

                # Create the header line
                header_line_dummy = ' '.join(key.center(column_widths_dummy[key]) for key in dummy_data.keys())
                html += header_line_dummy + "\n"

                # Create the separator line
                separator_dummy = ' '.join('-' * column_widths_dummy[key] for key in dummy_data.keys())
                html += separator_dummy + "\n"

                # Format each row of the dummy data
                for i in range(len(dummy_data['Variable'])):
                    row = []
                    for key in dummy_data.keys():
                        if key == 'Conf. Interval':
                            ci_text = f"{dummy_data[key][i][0]} - {dummy_data[key][i][1]}"
                            row.append(ci_text.center(column_widths_dummy[key]))
                        else:
                            value = str(dummy_data[key][i])
                            row.append(value.center(column_widths_dummy[key]))
                    html += ' '.join(row) + "\n"
                html += "</pre>\n"

        return display(HTML(html))



