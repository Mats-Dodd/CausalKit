import numpy as np
import pandas as pd
import scipy.stats as stats
from IPython.display import display, HTML

CRITICAL = 1.96


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
        self.regressors = independent
        self.num_regressors = len(independent)
        self.observations = len(self.data)
        self.degrees_of_freedom = self.observations - self.num_regressors - (1 if self.intercept else 0)
        self.coefficients = None
        self.rss = None
        self.standard_errors = None
        self.t_stats = None
        self.p_values = None
        self.conf_int = None
        self.summary_data = None
        self.table = None

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
        self._residual_sum_of_squares()
        self._fit_standard_errors()
        self._fit_t_statistic()
        self._fit_p_values()
        self._fit_conf_int()
        self.set_summary_data()
        self.set_table()

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

    def _residual_sum_of_squares(self):
        self.rss = np.sum(self.residuals() ** 2)

    def _fit_standard_errors(self):
        error_variance = self.rss / self.degrees_of_freedom
        xtx_inv = np.linalg.inv(self.independent.T @ self.independent)

        if self.standard_error_type == 'non-robust':
            self.standard_errors = np.sqrt(np.diag(xtx_inv * error_variance))

        elif self.standard_error_type == 'robust':
            return np.sqrt(np.diag(np.linalg.inv(self.independent.T @ self.independent) @ self.independent.T @ np.diag(self.residuals() ** 2) @ self.independent))

        elif self.standard_error_type == 'clustered':
            """TODO: Implement clustered standard errors"""
            pass

    def _fit_t_statistic(self):

        self.t_stats = self.coefficients / self.standard_errors

    def _fit_p_values(self):

        self.p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_stats), self.degrees_of_freedom))

    def _fit_conf_int(self):

        lower_bounds = self.coefficients - CRITICAL * self.standard_errors
        upper_bounds = self.coefficients + CRITICAL * self.standard_errors
        self.conf_int = [[lower, upper] for lower, upper in zip(lower_bounds, upper_bounds)]

    def set_summary_data(self):
        self.summary_data = {
            'Variable': ['Intercept'] + self.regressors,
            'Coefficient': [round(num, 3) for num in self.coefficients],
            'Std-Error': [round(num, 3) for num in self.standard_errors],
            'T-Statistic': [round(num, 3) for num in self.t_stats],
            'P>|t|': [round(num, 3) for num in self.p_values],
            'Conf. Interval': [[round(num, 3) for num in sublist] for sublist in self.conf_int]
        }

    def set_table(self):
        data = self.summary_data.copy()
        data['Lower Bound'] = [ci[0] for ci in data['Conf. Interval']]
        data['Upper Bound'] = [ci[1] for ci in data['Conf. Interval']]
        del data['Conf. Interval']

        self.table = pd.DataFrame(data)

    def summary(self, content_type='dynamic'):

        summary_data = self.summary_data

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

        if content_type == 'dynamic':
            html = "<h2 style='text-align: center;'>Regression Results</h2><hr style='border-style: dashed;'>"

            html += "<div class='container'>"

            for key in summary_data.keys():
                # Column container
                html += "<div class='column'>"

                # Header
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

                for value in summary_data[key]:
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
        else:
            html = "<h1 style='text-align:center;'>Regression Results</h1><pre style='text-align:center; font-family:monospace;'>"

            column_widths = {key: max(max([len(str(x)) for x in summary_data[key]]), len(key)) for key in summary_data.keys()}
            column_widths['Conf. Interval'] = max(max([len(f"{x[0]} - {x[1]}") for x in summary_data['Conf. Interval']]), len('Conf. Interval'))

            headers = [key.center(column_widths[key]) for key in summary_data.keys()]
            html += ' '.join(headers) + "\n"

            separator = '-'.join('-' * column_widths[key] for key in summary_data.keys())
            html += separator + "\n"

            for i in range(len(summary_data['Variable'])):
                row = []
                for key in summary_data.keys():
                    if key == 'Conf. Interval':

                        ci_text = f"{summary_data[key][i][0]} - {summary_data[key][i][1]}"
                        row.append(ci_text.center(column_widths[key]))
                    else:
                        row.append(str(summary_data[key][i]).center(column_widths[key]))
                html += ' '.join(row) + "\n"

            html += "</pre>"

        return display(HTML(html))
