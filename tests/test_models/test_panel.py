from src.models.panel import FixedEffects

import numpy as np
import pandas as pd
from src.models.linreg import LinReg
import statsmodels.api as sm
import pytest


@pytest.fixture(scope="class")
def panel_data():
    np.random.seed(69)
    ids = range(1, 6)  # Assuming 5 different entities
    years = range(2011, 2016)  # Assuming 5 different years

    data_list = []
    for id in ids:
        for year in years:
            x = np.random.uniform(0, 10)
            y = 2 * x + 1 + np.random.normal(0, 1)
            data_list.append({'id': id, 'year': year, 'outcome': y, 'independent': x})

    panel_data = pd.DataFrame(data_list)
    return panel_data


@pytest.fixture(scope="class")
def fe_model(panel_data):
    return FixedEffects(df=panel_data,
                        outcome='outcome',
                        independent=['independent'],
                        fixed=['year'])


class TestFixedEffects:

    def test_model(self, fe_model):
        m = fe_model
        assert isinstance(m, FixedEffects)




#%%
