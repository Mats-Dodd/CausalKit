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
        assert np.isclose(round(linreg_model_1.predict(1)[0], 2), round(2.659, 2))

    def test_coefficients(self, linreg_model_1, sm_model_1):
        assert np.isclose(linreg_model_1.coefficients[0], sm_model_1.params.iloc[0])
        assert np.isclose(linreg_model_1.coefficients[1], sm_model_1.params.iloc[1])

    def test_standard_errors(self, linreg_model_1, sm_model_1):
        assert np.isclose(linreg_model_1.standard_errors[0], sm_model_1.bse.iloc[0])
        assert np.isclose(linreg_model_1.standard_errors[1], sm_model_1.bse.iloc[1])

    def test_p_values(self, linreg_model_1, sm_model_1):
        assert np.isclose(linreg_model_1.p_values[0], sm_model_1.pvalues.iloc[0])
        assert np.isclose(linreg_model_1.p_values[1], sm_model_1.pvalues.iloc[1])


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
        assert np.isclose(linreg_model_2.coefficients[0], sm_model_2.params.iloc[0])
        assert np.isclose(linreg_model_2.coefficients[1], sm_model_2.params.iloc[1])
        assert np.isclose(linreg_model_2.coefficients[2], sm_model_2.params.iloc[2])
