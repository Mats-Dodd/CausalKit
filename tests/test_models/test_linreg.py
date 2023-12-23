import numpy as np
import pandas as pd
from src.models.linreg import LinReg

np.random.seed(69)
independent_values = np.random.normal(0, 3, 1000)
outcome = 2 * independent_values + np.random.normal(0, 1, 1000)
data = pd.DataFrame()
data = data.assign(outcome=outcome, independent=independent_values)


def test_predict():

    lm1 = LinReg(df=data, outcome='outcome', independent=['independent'])

    assert np.isclose(round(lm1.predict(1)[0], 2), round(2.01, 2))

#%%
