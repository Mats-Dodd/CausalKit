import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PermInf:

    """
    Nonparametric Permutation Test

    """

    def __init__(self,
                 df: pd.DataFrame,
                 treatment_column: str,
                 outcome_column: str,
                 function=None,
                 simulations: int = 1000,
                 show_results=True):
        """
        Parameters
        ----------
        df: Pandas Dataframe object containing data of interest

        treatment_column: String of treatments column name

        outcome_column: String of the outcome of interest column name

        function: Function of interest to compare groups, if none provided, difference in means computed

        simulations: The number of permutation samples used to compute our p value
        """
        self.data = df
        self.simulations = simulations
        self.treat = treatment_column
        self.out = outcome_column
        self.function = function
        self.results = []
        self.test_statistic = None
        self.test_statistic_p_value = None
        self.show_results = show_results

    def fit(self):

        if self.function is None:

            self.function = lambda x, y: abs(np.mean(y) - np.mean(x))

        treatment_group_outcomes = self.data[(self.data[self.treat] == 1) | (self.data[self.treat] == 1.0)][self.out]

        control_group_outcomes = self.data[(self.data[self.treat] == 0) | (self.data[self.treat] == 0.0)][self.out]

        self.test_statistic = self.function(treatment_group_outcomes, control_group_outcomes)

        for iteration in range(self.simulations):
            df = pd.DataFrame(self.data[self.out])
            random_assignment = np.random.permutation(self.data[self.treat])
            df = df.assign(assignment=random_assignment)
            treatment_group_outcomes = df[df['assignment'] == 1][self.out]
            control_group_outcomes = df[df['assignment'] == 0][self.out]
            z = self.function(treatment_group_outcomes, control_group_outcomes)
            self.results.append(z)

        number_greater_or_equal = np.greater_equal(self.test_statistic, np.array(self.results))
        number_less_or_equal = np.less_equal(self.test_statistic, np.array(self.results))

        count_greater = 0
        for result in self.results:
            if result >= self.test_statistic:
                count_greater += 1

        self.test_statistic_p_value = count_greater / self.simulations

        if self.show_results:
            print(f" Test Statistic: {round(self.test_statistic,3)}  \n P-Value: {self.test_statistic_p_value}")

    def summarize(self):

        plt.hist(self.results, bins=50)
        plt.axvline(x=self.test_statistic, color='r')
        plt.title("Permutation distribution of test statistic")
        plt.xlabel("Value of Statistic")
        plt.ylabel("Frequency")
        plt.show()


class BootstrapCI:
    """
    Nonparametric Computation of Estimator Confidence Intervals using the Bootstrapping Technique

    """

    def __init__(self,
                 df: pd.DataFrame,
                 treatment_column: str,
                 outcome_column: str,
                 function=None,
                 simulations: int = 1000,
                 alpha: float = 0.95):
        """
        Parameters
        ----------
        df: Pandas Dataframe object containing data of interest

        treatment_column: String of treatments column name

        outcome_column: String of the outcome of interest column name

        function: Function of interest to compare groups, if none provided, difference in means computed

        simulations: The number of permutation samples used to compute our p value

        alpha: confidence level for computed confidence intervals
        """
        self.data = df
        self.simulations = simulations
        self.treat = treatment_column
        self.out = outcome_column
        self.function = function
        self.simulations = simulations
        self.alpha = alpha
        self.test_statistic = None
        self.results = []
        self.standard_error = None
        self.upper_bound = None
        self.lower_bound = None

    def fit(self):

        if self.function is None:

            self.function = lambda x, y: abs(np.mean(y) - np.mean(x))

        treatment_group_outcomes = self.data[(self.data[self.treat] == 1) | (self.data[self.treat] == 1.0)][self.out]

        control_group_outcomes = self.data[(self.data[self.treat] == 0) | (self.data[self.treat] == 0.0)][self.out]

        self.test_statistic = self.function(treatment_group_outcomes, control_group_outcomes)

        control_data = self.data[(self.data[self.treat] == 0) | (self.data[self.treat] == 0.0)]
        treatment_data = self.data[(self.data[self.treat] == 1) | (self.data[self.treat] == 1.0)]

        nobs_control = len(control_data)
        nobs_treatment = len(treatment_data)

        for iteration in range(self.simulations):

            control_group_sample = control_data.sample(nobs_control, replace=True)[self.out]
            treatment_group_sample = treatment_data.sample(nobs_treatment, replace=True)[self.out]

            z = self.function(treatment_group_sample, control_group_sample)

            self.results.append(z)

        self.standard_error = np.std(self.results)

        level = 1 - self.alpha

        self.lower_bound = np.quantile(self.results, 1-level/2)
        self.upper_bound = np.quantile(self.results, level/2)

        print(f" Test Statistic: {round(self.test_statistic,3)}  \n Standard Error: {self.standard_error}"
              f"\n CI: {self.lower_bound, self.upper_bound}")

    def summarize(self):

        plt.hist(self.results, bins=50)
        plt.axvline(x=self.test_statistic, color='r')
        plt.title("Bootstrap distribution of test statistic")
        plt.xlabel("Value of Statistic")
        plt.ylabel("Frequency")
        plt.show()


class Power:

    def __init__(self,
                 control_mean,
                 treatment_mean,
                 control_sd,
                 treatment_sd,
                 observations: int = 1000,
                 experiment_simulations: int = 20,
                 perminf_loops: int = 1000):

        self.control_mean = control_mean
        self.treatment_mean = treatment_mean
        self.control_sd = control_sd
        self.treatment_sd = treatment_sd
        self.observations = observations
        self.experiment_simulations = experiment_simulations
        self.perminf_loops = perminf_loops
        self.p_values = ()
        self.power_value = None

    def fit(self):

        self.p_values = []

        for experiment in range(self.experiment_simulations):

            control_values = np.random.normal(self.control_mean, self.control_sd, self.observations)
            control_treatment = np.zeros(self.observations)
            treatment_values = np.random.normal(self.treatment_mean, self.treatment_sd, self.observations)
            treatment_treatment = np.ones(self.observations)
            values = np.hstack((control_values, treatment_values))
            assignment = np.hstack((control_treatment, treatment_treatment))
            data = np.stack((assignment, values), axis=-1)
            cols = ['assignment', 'values']
            df = pd.DataFrame(data=data, columns=cols)

            experiment_model = PermInf(df=df,
                                       treatment_column='assignment',
                                       outcome_column='values',
                                       simulations=self.perminf_loops,
                                       show_results=False)
            experiment_model.fit()
            self.p_values.append(experiment_model.test_statistic_p_value)

        count_significant = 0
        for p_value in self.p_values:
            if p_value <= 0.05:
                count_significant += 1

        self.power_value = count_significant/self.experiment_simulations

        print(f"Experiment Power: {self.power_value}")

    def summarize(self):
        plt.hist(self.p_values, bins=25)
        plt.title("Simulated Experiment P Values")
        plt.xlabel("P Values")
        plt.ylabel("Frequency")
        plt.show()