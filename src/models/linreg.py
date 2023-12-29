import numpy as np
import pandas as pd
import scipy.stats as stats
from IPython.display import display, HTML


class LinReg:

    """ Class to represent a Linear Regression model"""

    def __init__(self,
                 df: pd.DataFrame,
                 outcome: str,
                 independent: list,
                 intercept: bool = True,
                 standard_error_type: str = ['non-robust', 'robust', 'clustered'][0]):
        self.data = df
        self.outcome = outcome
        self.independent_vars = independent
        self.independent_data = self.data[self.independent_vars].values
        self.dependent_data = self.data[self.outcome].values
        self.intercept = intercept
        self.standard_error_type = standard_error_type
        self.n_regs = len(independent)
        self.obs = len(self.data)
        self.degrees_of_freedom = self.obs - self.n_regs - (1 if self.intercept else 0)
        self.k = self.n_regs + (1 if self.intercept else 0)
        self.coefficients = None
        self.rss = None
        self.tss = None
        self.r_squared = None
        self.adj_r_squared = None
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        self.adj_aic = None
        self.adj_bic = None
        self.f_stat = None
        self.f_stat_p_value = None
        self.standard_errors = None
        self.t_stats = None
        self.p_values = None
        self.conf_int = None
        self.summary_data_coefficients = None
        self.summary_data_model = None
        self.table = None

        self._fit()

    def _add_intercept(self, x):
        if self.intercept:

            if x.ndim == 1:
                x = x.reshape(-1, 1)
            return np.c_[np.ones((x.shape[0], 1)), x]
        return x

    def _fit_coefficients(self):
        x = self.independent_data
        if self.intercept:
            x = self._add_intercept(x)
        y = self.dependent_data

        self.coefficients = np.linalg.lstsq(x, y, rcond=None)[0]

    def _fit(self):
        self._fit_coefficients()
        self._residual_sum_of_squares()
        self._total_sum_of_squares()
        self._r_squared()
        self._adjusted_r_squared()
        self._log_likelihood()
        self._aic()
        self._bic()
        self._adjusted_aic()
        self._adjusted_bic()
        self.f_statistic()
        self.f_statistic_p_value()
        self._fit_standard_errors()
        self._fit_t_statistic()
        self._fit_p_values()
        self._fit_conf_int()
        self.set_summary_data()
        self.set_table()

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

    def _residual_sum_of_squares(self):
        self.rss = np.sum(self.residuals() ** 2)

    def _total_sum_of_squares(self):
        self.tss = np.sum((self.dependent_data - np.mean(self.dependent_data)) ** 2)

    def _r_squared(self):
        self.r_squared = 1 - self.rss / self.tss

    def _adjusted_r_squared(self):
        self.adj_r_squared = 1 - (1 - self.r_squared) * (self.obs - 1) / self.degrees_of_freedom

    def _log_likelihood(self):
        n = self.obs
        rss = self.rss
        k = self.k
        self.log_likelihood = -n/2 * np.log(2 * np.pi) - n/2 * np.log(rss / (n - k)) - rss / (2 * (rss / (n - k)))

    def _aic(self):
        k = self.k
        self.aic = -2 * self.log_likelihood + 2 * k

    def _bic(self):
        k = self.n_regs + (1 if self.intercept else 0)
        n = self.obs
        self.bic = -2 * self.log_likelihood + np.log(n) * k

    def _adjusted_aic(self):
        self.adj_aic = self.aic + 2 * (self.n_regs + 1) * (self.n_regs + 2) / (self.obs - self.n_regs - 2)

    def _adjusted_bic(self):
        bic = self.bic
        self.adj_bic = bic + np.log(self.obs) * (self.n_regs + 1) * (self.n_regs + 2) / (self.obs - self.n_regs - 2)

    def f_statistic(self):
        self.f_stat = (self.tss - self.rss) / self.n_regs / (self.rss / self.degrees_of_freedom)

    def f_statistic_p_value(self):
        p = self.n_regs
        df1 = p
        df2 = self.obs - p - (1 if self.intercept else 0)
        self.f_stat_p_value = 1 - stats.f.cdf(self.f_stat, df1, df2)

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

        elif self.standard_error_type == 'clustered':
            """TODO: Implement clustered standard errors"""
            pass

    def _fit_t_statistic(self):

        self.t_stats = self.coefficients / self.standard_errors

    def _fit_p_values(self):

        self.p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_stats), self.degrees_of_freedom))

    def _fit_conf_int(self):
        df = self.degrees_of_freedom
        confidence = 0.95
        t_crit = stats.t.ppf((1 + confidence) / 2, df)

        lower_bounds = self.coefficients - t_crit * self.standard_errors
        upper_bounds = self.coefficients + t_crit * self.standard_errors

        self.conf_int = [[lower, upper] for lower, upper in zip(lower_bounds, upper_bounds)]

    def set_summary_data(self):
        self.summary_data_coefficients = {
            'Variable': ['Intercept'] + self.independent_vars,
            'Coefficient': [round(num, 3) for num in self.coefficients],
            'Std-Error': [round(num, 3) for num in self.standard_errors],
            'T-Statistic': [round(num, 3) for num in self.t_stats],
            'P>|t|': [round(num, 3) for num in self.p_values],
            'Conf. Interval': [[round(num, 3) for num in sublist] for sublist in self.conf_int]
        }

        self.summary_data_model = {
            'Dep. Variable': self.outcome,
            'Observations': self.obs,
            'Standard Error Type': self.standard_error_type,
            'R-squared': round(self.r_squared, 3),
            'Adj. R-squared': round(self.adj_r_squared, 3),
            'Log-Likelihood': round(self.log_likelihood, 3),
            'AIC': round(self.aic, 3),
            'BIC': round(self.bic, 3),
            'Adj. AIC': round(self.adj_aic, 3),
            'Adj. BIC': round(self.adj_bic, 3),
            'F-statistic': round(self.f_stat, 3),
            'Prob (F-statistic)': round(self.f_stat_p_value, 3)

        }

    def set_table(self):
        data = self.summary_data_coefficients.copy()
        data['Lower Bound'] = [ci[0] for ci in data['Conf. Interval']]
        data['Upper Bound'] = [ci[1] for ci in data['Conf. Interval']]
        del data['Conf. Interval']

        self.table = pd.DataFrame(data)

    def summary(self, content_type='dynamic'):

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
            html = "<h1 style='text-align:center;'>Regression Results</h1><pre style='text-align:center; font-family:monospace;'>"

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
