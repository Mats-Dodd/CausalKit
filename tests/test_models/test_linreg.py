import numpy as np
import pandas as pd
from src.models.linreg import LinReg
import statsmodels.api as sm

np.random.seed(69)
x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.normal(0, 1, 100)
data = pd.DataFrame()
data = data.assign(outcome=y, independent=x)


def test_predict():

    model = LinReg(df=data,
                   outcome="outcome",
                   independent=["independent"])

    assert np.isclose(round(model.predict(1)[0], 2), round(2.659, 2))


def test_coefficients():

    model = LinReg(df=data,
                   outcome="outcome",
                   independent=["independent"])

    model2 = sm.OLS(data["outcome"], sm.add_constant(data["independent"])).fit()

    assert np.isclose(model.coefficients[0], model2.params[0])
    assert np.isclose(model.coefficients[1], model2.params[1])


def test_standard_errors():

    model = LinReg(df=data,
                   outcome="outcome",
                   independent=["independent"])

    model2 = sm.OLS(data["outcome"], sm.add_constant(data["independent"])).fit()

    assert np.isclose(model.standard_errors[0], model2.bse[0])
    assert np.isclose(model.standard_errors[1], model2.bse[1])

def test_p_values():

    model = LinReg(df=data,
                   outcome="outcome",
                   independent=["independent"])

    model2 = sm.OLS(data["outcome"], sm.add_constant(data["independent"])).fit()

    assert np.isclose(model.p_values[0], model2.pvalues[0])
    assert np.isclose(model.p_values[1], model2.pvalues[1])
#%%
