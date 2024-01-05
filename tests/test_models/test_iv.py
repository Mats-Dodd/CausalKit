import numpy as np
import pandas as pd
import pytest

from src.models.linreg import LinReg
from src.models.iv import IV


@pytest.fixture(scope="class")
def iv_regression_data_basic():
    np.random.seed(69)
    x = np.linspace(0, 10, 100)
    z = x + np.random.normal(0, 1, 100)
    y = 3 * x + np.random.normal(0, 1, 100)
    data = pd.DataFrame({'outcome': y, 'independent': x, 'instrument': z})
    return data


@pytest.fixture(scope="class")
def iv_model_1(iv_regression_data_basic):
    return IV(df=iv_regression_data_basic,
              outcome='outcome',
              independent=['independent'],
              controls=[],
              instruments=['instrument'])


@pytest.fixture(scope="class")
def two_stage_model(iv_regression_data_basic):
    first_stage = LinReg(df=iv_regression_data_basic,
                         outcome='independent',
                         independent=['instrument'])
    predicted = first_stage.predict(iv_regression_data_basic['instrument'].values)
    iv_regression_data_basic = iv_regression_data_basic.assign(independent_hat=predicted)

    second_stage = LinReg(df=iv_regression_data_basic,
                          outcome='outcome',
                          independent=['independent_hat'])
    return second_stage


class TestSimpleIV:

    def test_iv_initialization(self, iv_model_1):
        assert iv_model_1.outcome == 'outcome'
        assert iv_model_1.independent_vars == ['independent']
        assert iv_model_1.controls == []
        assert iv_model_1.instruments == ['instrument']

    def test_first_stage_regression(self, iv_model_1):

        assert iv_model_1.first_stage_model is not None

    def test_second_stage_regression(self, iv_model_1):

        assert iv_model_1.second_stage_model is not None

    def test_iv_model_coefficients(self, iv_model_1, two_stage_model):
        assert np.isclose(iv_model_1.coefficients[0], two_stage_model.coefficients[0], atol=1e-2)
        assert np.isclose(iv_model_1.coefficients[1], two_stage_model.coefficients[1], atol=1e-2)

    def test_iv_model_standard_errors(self, iv_model_1, two_stage_model):
        assert np.isclose(iv_model_1.standard_errors[0], two_stage_model.standard_errors[0], atol=1e-2)
        assert np.isclose(iv_model_1.standard_errors[1], two_stage_model.standard_errors[1], atol=1e-2)


@pytest.fixture(scope="class")
def iv_regression_data_two_instruments():
    np.random.seed(69)
    x = np.linspace(0, 10, 100)
    z1 = x + np.random.normal(0, 1, 100)
    z2 = 0.5 * x + np.random.normal(0, 1.5, 100)
    y = 3 * x + np.random.normal(0, 1, 100)
    data = pd.DataFrame({'outcome': y, 'independent': x, 'instrument1': z1, 'instrument2': z2})
    return data


@pytest.fixture(scope="class")
def iv_model_2(iv_regression_data_two_instruments):
    return IV(df=iv_regression_data_two_instruments,
              outcome='outcome',
              independent=['independent'],
              controls=[],
              instruments=['instrument1', 'instrument2'])


@pytest.fixture(scope="class")
def two_stage_model_2(iv_regression_data_two_instruments):
    first_stage = LinReg(df=iv_regression_data_two_instruments,
                         outcome='independent',
                         independent=['instrument1', 'instrument2'])
    predicted = first_stage.predict(iv_regression_data_two_instruments[['instrument1', 'instrument2']].values)
    iv_regression_data_two_instruments = iv_regression_data_two_instruments.assign(independent_hat=predicted)

    second_stage = LinReg(df=iv_regression_data_two_instruments,
                          outcome='outcome',
                          independent=['independent_hat'])
    return second_stage


class TestTwoInstrumentsIV:

    def test_iv_initialization(self, iv_model_2):
        assert iv_model_2.outcome == 'outcome'
        assert iv_model_2.independent_vars == ['independent']
        assert iv_model_2.controls == []
        assert iv_model_2.instruments == ['instrument1', 'instrument2']

    def test_first_stage_regression(self, iv_model_2):

        assert iv_model_2.first_stage_model is not None

    def test_second_stage_regression(self, iv_model_2):

        assert iv_model_2.second_stage_model is not None

    def test_iv_model_coefficients(self, iv_model_2, two_stage_model_2):
        assert np.isclose(iv_model_2.coefficients[0], two_stage_model_2.coefficients[0], atol=1e-2)
        assert np.isclose(iv_model_2.coefficients[0], two_stage_model_2.coefficients[0], atol=1e-2)

    def test_iv_model_standard_errors(self, iv_model_2, two_stage_model_2):
        assert np.isclose(iv_model_2.standard_errors[0], two_stage_model_2.standard_errors[0], atol=1e-2)
        assert np.isclose(iv_model_2.standard_errors[1], two_stage_model_2.standard_errors[1], atol=1e-2)


@pytest.fixture(scope="class")
def iv_regression_data_with_controls():
    np.random.seed(69)
    x = np.linspace(0, 10, 100)
    z1 = x + np.random.normal(0, 1, 100)
    z2 = 0.5 * x + np.random.normal(0, 1.5, 100)
    control1 = np.random.normal(5, 2, 100)
    control2 = np.random.normal(-3, 1, 100)
    y = 3 * x + 1.5 * control1 - 2 * control2 + np.random.normal(0, 1, 100)  # Outcome variable
    data = pd.DataFrame({
        'outcome': y,
        'independent': x,
        'instrument1': z1,
        'instrument2': z2,
        'control1': control1,
        'control2': control2
    })
    return data


@pytest.fixture(scope="class")
def iv_model_3(iv_regression_data_with_controls):
    return IV(df=iv_regression_data_with_controls,
              outcome='outcome',
              independent=['independent'],
              controls=['control1', 'control2'],
              instruments=['instrument1', 'instrument2'])


@pytest.fixture(scope="class")
def two_stage_model_3(iv_regression_data_with_controls):
    first_stage = LinReg(df=iv_regression_data_with_controls,
                         outcome='independent',
                         independent=['instrument1', 'instrument2', 'control1', 'control2'])
    predicted = first_stage.predict(iv_regression_data_with_controls[['instrument1',
                                                                      'instrument2',
                                                                      'control1',
                                                                      'control2']].values)
    iv_regression_data_with_controls = iv_regression_data_with_controls.assign(independent_hat=predicted)

    second_stage = LinReg(df=iv_regression_data_with_controls,
                          outcome='outcome',
                          independent=['independent_hat', 'control1', 'control2'])
    return second_stage


class TestIVWithControls:

    def test_iv_initialization(self, iv_model_3):
        assert iv_model_3.outcome == 'outcome'
        assert iv_model_3.independent_vars == ['independent']
        assert iv_model_3.controls == ['control1', 'control2']
        assert iv_model_3.instruments == ['instrument1', 'instrument2']

    def test_first_stage_regression(self, iv_model_3):

        assert iv_model_3.first_stage_model is not None

    def test_second_stage_regression(self, iv_model_3):

        assert iv_model_3.second_stage_model is not None

    def test_iv_model_coefficients(self, iv_model_3, two_stage_model_3):
        assert np.isclose(iv_model_3.coefficients[0], two_stage_model_3.coefficients[0], atol=1e-2)
        assert np.isclose(iv_model_3.coefficients[1], two_stage_model_3.coefficients[1], atol=1e-2)
        assert np.isclose(iv_model_3.coefficients[2], two_stage_model_3.coefficients[2], atol=1e-2)
        assert np.isclose(iv_model_3.coefficients[3], two_stage_model_3.coefficients[3], atol=1e-2)

    def test_iv_model_standard_errors(self, iv_model_3, two_stage_model_3):
        assert np.isclose(iv_model_3.standard_errors[0], two_stage_model_3.standard_errors[0], atol=1e-2)
        assert np.isclose(iv_model_3.standard_errors[1], two_stage_model_3.standard_errors[1], atol=1e-2)
        assert np.isclose(iv_model_3.standard_errors[2], two_stage_model_3.standard_errors[2], atol=1e-2)
        assert np.isclose(iv_model_3.standard_errors[3], two_stage_model_3.standard_errors[3], atol=1e-2)
