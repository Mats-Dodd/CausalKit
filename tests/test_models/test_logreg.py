import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pytest

from src.models.logreg import LogReg


@pytest.fixture(scope="class")
def logistic_regression_data():
    df = sm.datasets.spector.load_pandas().data
    return df


@pytest.fixture(scope="class")
def logreg_model(logistic_regression_data):
    return LogReg(df=logistic_regression_data,
                  outcome="GRADE",
                  independent=["GPA", "TUCE", "PSI"])


@pytest.fixture(scope="class")
def sm_model(logistic_regression_data):
    return smf.logit("GRADE ~ GPA + TUCE + PSI", data=logistic_regression_data).fit()


class TestLogisticRegression:

    def test_predict_(self, logreg_model):
        assert np.isclose(logreg_model.predict(np.array([2.66, 20, 0])), 0.026578, atol=1e-2)

    def test_predict_class(self, logreg_model):
        assert np.isclose(logreg_model.predict_class(np.array([2.66, 20, 0])), 0, atol=1e-2)

    def test_coefficients(self, logreg_model, sm_model):
        assert np.isclose(logreg_model.coefficients[0], sm_model.params.iloc[0], atol=1e-2)
        assert np.isclose(logreg_model.coefficients[1], sm_model.params.iloc[1], atol=1e-2)
        assert np.isclose(logreg_model.coefficients[2], sm_model.params.iloc[2], atol=1e-2)
        assert np.isclose(logreg_model.coefficients[3], sm_model.params.iloc[3], atol=1e-2)

    def test_standard_errors(self, logreg_model, sm_model):
        assert np.isclose(logreg_model.standard_errors[0], sm_model.bse.iloc[0], atol=1e-2)
        assert np.isclose(logreg_model.standard_errors[1], sm_model.bse.iloc[1], atol=1e-2)
        assert np.isclose(logreg_model.standard_errors[2], sm_model.bse.iloc[2], atol=1e-2)
        assert np.isclose(logreg_model.standard_errors[3], sm_model.bse.iloc[3], atol=1e-2)

    def test_fitted_values(self, logreg_model, sm_model):

        assert np.allclose(logreg_model.fitted_values(), sm_model.predict(), atol=1e-2)

