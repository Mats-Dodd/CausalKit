import numpy as np
import pandas as pd
from src.models.linreg import LinReg
import statsmodels.api as sm
import pytest


@pytest.fixture(scope="class")
def regression_data_1():
    np.random.seed(69)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.normal(0, 1, 100)
    data = pd.DataFrame({'outcome': y, 'independent': x})
    return data


@pytest.fixture(scope="class")
def linreg_model_1(regression_data_1):
    return LinReg(df=regression_data_1, outcome="outcome", independent=["independent"])


@pytest.fixture(scope="class")
def sm_model_1(regression_data_1):
    return sm.OLS(regression_data_1["outcome"], sm.add_constant(regression_data_1["independent"])).fit()


class TestSimpleRegression:

    def test_predict_sls(self, linreg_model_1):
        assert np.isclose(round(linreg_model_1.predict(1)[0], 2), round(2.659, 2), atol=1e-2)

    def test_coefficients(self, linreg_model_1, sm_model_1):
        assert np.isclose(linreg_model_1.coefficients[0], sm_model_1.params.iloc[0], atol=1e-2)
        assert np.isclose(linreg_model_1.coefficients[1], sm_model_1.params.iloc[1], atol=1e-2)

    def test_standard_errors(self, linreg_model_1, sm_model_1):
        assert np.isclose(linreg_model_1.standard_errors[0], sm_model_1.bse.iloc[0], atol=1e-2)
        assert np.isclose(linreg_model_1.standard_errors[1], sm_model_1.bse.iloc[1], atol=1e-2)

    def test_p_values(self, linreg_model_1, sm_model_1):
        assert np.isclose(linreg_model_1.p_values[0], sm_model_1.pvalues.iloc[0], atol=1e-2)
        assert np.isclose(linreg_model_1.p_values[1], sm_model_1.pvalues.iloc[1], atol=1e-2)

    def test_t_statistics(self, linreg_model_1, sm_model_1):
        assert np.isclose(linreg_model_1.t_stats[0], sm_model_1.tvalues.iloc[0], atol=1e-2)
        assert np.isclose(linreg_model_1.t_stats[1], sm_model_1.tvalues.iloc[1], atol=1e-2)

    def test_confint(self, linreg_model_1, sm_model_1):
        assert np.isclose(linreg_model_1.conf_int[0][0], sm_model_1.conf_int().iloc[0, 0], atol=1e-2)
        assert np.isclose(linreg_model_1.conf_int[0][1], sm_model_1.conf_int().iloc[0, 1], atol=1e-2)
        assert np.isclose(linreg_model_1.conf_int[1][0], sm_model_1.conf_int().iloc[1, 0], atol=1e-2)
        assert np.isclose(linreg_model_1.conf_int[1][1], sm_model_1.conf_int().iloc[1, 1], atol=1e-2)


@pytest.fixture(scope="class")
def regression_data_2():
    np.random.seed(69)
    x = np.linspace(0, 5, 50)
    z = np.linspace(3, 8, 50)
    y = 1 + 3 * x + 5*z + np.random.normal(0, 2, 50)
    data = pd.DataFrame({'outcome': y,
                         'independent1': x,
                         'independent2': z})
    return data


@pytest.fixture(scope="class")
def linreg_model_2(regression_data_2):
    return LinReg(df=regression_data_2, outcome="outcome", independent=["independent1", "independent2"])


@pytest.fixture(scope="class")
def sm_model_2(regression_data_2):
    x = regression_data_2[['independent1', 'independent2']]
    y = regression_data_2['outcome']
    x = sm.add_constant(x)
    return sm.OLS(y, x).fit()


class TestMultipleRegression:

    def test_coefficients(self, linreg_model_2, sm_model_2):
        assert np.isclose(linreg_model_2.coefficients[0], sm_model_2.params.iloc[0], atol=1e-2)
        assert np.isclose(linreg_model_2.coefficients[1], sm_model_2.params.iloc[1], atol=1e-2)
        assert np.isclose(linreg_model_2.coefficients[2], sm_model_2.params.iloc[2], atol=1e-2)


@pytest.fixture(scope="class")
def regression_data_3():
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 3 * x + 5 + np.random.normal(0, 5, 100)
    data = pd.DataFrame({'outcome': y, 'independent': x})
    return data


@pytest.fixture(scope="class")
def linreg_model_3(regression_data_3):
    return LinReg(df=regression_data_3, outcome="outcome", independent=["independent"])


@pytest.fixture(scope="class")
def sm_model_3(regression_data_3):
    return sm.OLS(regression_data_3["outcome"], sm.add_constant(regression_data_3["independent"])).fit()


class TestRegressionWithNoise:

    def test_coefficients_noise(self, linreg_model_3, sm_model_3):

        assert np.isclose(linreg_model_3.coefficients[0], sm_model_3.params.iloc[0], atol=1e-2)
        assert np.isclose(linreg_model_3.coefficients[1], sm_model_3.params.iloc[1], atol=1e-2)

    def test_standard_errors_noise(self, linreg_model_3, sm_model_3):

        assert np.isclose(linreg_model_3.standard_errors[0], sm_model_3.bse.iloc[0], atol=1e-2)
        assert np.isclose(linreg_model_3.standard_errors[1], sm_model_3.bse.iloc[1], atol=1e-2)

    def test_p_values_noise(self, linreg_model_3, sm_model_3):

        assert np.isclose(linreg_model_3.p_values[0], sm_model_3.pvalues.iloc[0], atol=1e-2)
        assert np.isclose(linreg_model_3.p_values[1], sm_model_3.pvalues.iloc[1], atol=1e-2)

    def test_t_statistics_noise(self, linreg_model_3, sm_model_3):

        assert np.isclose(linreg_model_3.t_stats[0], sm_model_3.tvalues.iloc[0], atol=1e-2)
        assert np.isclose(linreg_model_3.t_stats[1], sm_model_3.tvalues.iloc[1], atol=1e-2)

    def test_confint_noise(self, linreg_model_3, sm_model_3):

        assert np.isclose(linreg_model_3.conf_int[0][0], sm_model_3.conf_int().iloc[0, 0], atol=1e-2)
        assert np.isclose(linreg_model_3.conf_int[0][1], sm_model_3.conf_int().iloc[0, 1], atol=1e-2)
        assert np.isclose(linreg_model_3.conf_int[1][0], sm_model_3.conf_int().iloc[1, 0], atol=1e-2)
        assert np.isclose(linreg_model_3.conf_int[1][1], sm_model_3.conf_int().iloc[1, 1], atol=1e-2)


@pytest.fixture(scope="class")
def regression_data_robust():
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.normal(0, x, 100)
    data = pd.DataFrame({'outcome': y, 'independent': x})
    return data


@pytest.fixture(scope="class")
def linreg_model_robust(regression_data_robust):
    model = LinReg(df=regression_data_robust,
                   outcome="outcome",
                   independent=["independent"],
                   standard_error_type='robust')
    return model


@pytest.fixture(scope="class")
def sm_model_robust(regression_data_robust):
    x = regression_data_robust["independent"]
    y = regression_data_robust["outcome"]
    X = sm.add_constant(x)
    return sm.OLS(y, X).fit(cov_type='HC0')


class TestRegressionRobust:

    def test_coefficients_robust(self, linreg_model_robust, sm_model_robust):
        assert np.isclose(linreg_model_robust.coefficients[0], sm_model_robust.params.iloc[0], atol=1e-2)
        assert np.isclose(linreg_model_robust.coefficients[1], sm_model_robust.params.iloc[1], atol=1e-2)

    def test_standard_errors_robust(self, linreg_model_robust, sm_model_robust):
        assert np.isclose(linreg_model_robust.standard_errors[0], sm_model_robust.bse.iloc[0], atol=1e-2)
        assert np.isclose(linreg_model_robust.standard_errors[1], sm_model_robust.bse.iloc[1], atol=1e-2)

    def test_p_values_robust(self, linreg_model_robust, sm_model_robust):
        assert np.isclose(linreg_model_robust.p_values[0], sm_model_robust.pvalues.iloc[0], atol=1e-2)
        assert np.isclose(linreg_model_robust.p_values[1], sm_model_robust.pvalues.iloc[1], atol=1e-2)

    def test_t_statistics_robust(self, linreg_model_robust, sm_model_robust):
        assert np.isclose(linreg_model_robust.t_stats[0], sm_model_robust.tvalues.iloc[0], atol=1e-2)
        assert np.isclose(linreg_model_robust.t_stats[1], sm_model_robust.tvalues.iloc[1], atol=1e-2)


@pytest.fixture(scope="class")
def regression_data_metrics():
    np.random.seed(0)
    x = np.linspace(0, 10, 100)
    y = 5 * x + 10 + np.random.normal(0, 2, 100)
    data = pd.DataFrame({'outcome': y, 'independent': x})
    return data


@pytest.fixture(scope="class")
def linreg_model_metrics(regression_data_metrics):
    return LinReg(df=regression_data_metrics, outcome="outcome", independent=["independent"])


@pytest.fixture(scope="class")
def sm_model_metrics(regression_data_metrics):
    x = regression_data_metrics["independent"]
    y = regression_data_metrics["outcome"]
    x = sm.add_constant(x)
    return sm.OLS(y, x).fit()


class TestRegressionMetrics:

    def test_r_squared(self, linreg_model_metrics, sm_model_metrics):
        assert np.isclose(linreg_model_metrics.r_squared, sm_model_metrics.rsquared, atol=1e-2)

    def test_adjusted_r_squared(self, linreg_model_metrics, sm_model_metrics):
        assert np.isclose(linreg_model_metrics.adj_r_squared, sm_model_metrics.rsquared_adj, atol=1e-2)

    def test_f_statistic(self, linreg_model_metrics, sm_model_metrics):
        assert np.isclose(linreg_model_metrics.f_stat, sm_model_metrics.fvalue, atol=1e-2)

    def test_f_p_value(self, linreg_model_metrics, sm_model_metrics):
        assert np.isclose(linreg_model_metrics.f_stat_p_value, sm_model_metrics.f_pvalue, atol=1e-3)

    def test_aic(self, linreg_model_metrics, sm_model_metrics):
        assert np.isclose(linreg_model_metrics.aic, sm_model_metrics.aic, atol=1e-1)

    def test_bic(self, linreg_model_metrics, sm_model_metrics):
        assert np.isclose(linreg_model_metrics.bic, sm_model_metrics.bic, atol=1e-1)




