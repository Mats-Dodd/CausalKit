from src.models.panel import FixedEffects

import numpy as np
import pandas as pd
from linearmodels import PanelOLS
import statsmodels.api as sm
import pytest


@pytest.fixture(scope="class")
def panel_data():
    np.random.seed(69)
    ids = range(1, 100)
    years = range(2010, 2021)
    industries = ['Tech', 'Health', 'Finance', 'Education',
                  'Retail', 'Energy', 'Manufacturing',
                  'Transport', 'Services', 'Agriculture']
    regions = ['North', 'South', 'East', 'West', 'Central']

    industry_effects = {'Tech': 3, 'Health': 2, 'Finance': 4,
                        'Education': 1, 'Retail': 2, 'Energy': 5, 'Manufacturing': 3,
                        'Transport': 4, 'Services': 2, 'Agriculture': 3}
    region_effects = {'North': 1, 'South': 2, 'East': 1, 'West': 3, 'Central': 2}

    data_list = []
    for id in ids:
        industry = industries[id % len(industries)]
        region = regions[id % len(regions)]
        for year in years:
            x = np.random.uniform(0, 10)
            industry_effect = industry_effects[industry]
            region_effect = region_effects[region]
            y = 1000 + 10 * x + 10 * year + 30 * id + 40 * region_effect + np.random.normal(0, 2)
            data_list.append({'id': id,
                              'year': year,
                              'industry': industry,
                              'region': region,
                              'outcome': y,
                              'independent': x})

    panel_data = pd.DataFrame(data_list)
    return panel_data


@pytest.fixture(scope="class")
def fe_model_one_way(panel_data):
    return FixedEffects(df=panel_data,
                        outcome='outcome',
                        independent=['independent'],
                        fixed=['year'],
                        standard_error_type='clustered')


@pytest.fixture(scope="class")
def fe_model_two_way(panel_data):
    return FixedEffects(df=panel_data,
                        outcome='outcome',
                        independent=['independent'],
                        fixed=['year', 'id'],
                        standard_error_type='two-way-clustered')


class TestFixedEffectsOneWay:

    def test_model(self, fe_model_one_way):
        m = fe_model_one_way
        assert isinstance(m, FixedEffects)

    def test_coefficients(self, fe_model_one_way, panel_data):
        m = fe_model_one_way
        panel_data = panel_data.set_index(['id', 'year'])
        model = PanelOLS.from_formula('outcome ~ independent + TimeEffects', data=panel_data)
        results = model.fit(cov_type='clustered', cluster_time=True)
        assert np.isclose(m.coefficients[1], results.params.iloc[0],  atol=1e-2)

    def test_standard_errors(self, fe_model_one_way, panel_data):
        m = fe_model_one_way
        panel_data = panel_data.set_index(['id', 'year'])
        model = PanelOLS.from_formula('outcome ~ independent + TimeEffects', data=panel_data)
        results = model.fit(cov_type='clustered', cluster_time=True)
        assert np.isclose(m.standard_errors[1], results.std_errors.iloc[0],  atol=1e-2)


class TestFixedEffectsTwoWay:

    def test_model(self, fe_model_two_way):
        m = fe_model_two_way
        assert isinstance(m, FixedEffects)

    def test_coefficients(self, fe_model_two_way, panel_data):
        m = fe_model_two_way
        panel_data = panel_data.set_index(['id', 'year'])
        model = PanelOLS.from_formula('outcome ~ independent + TimeEffects + EntityEffects', data=panel_data)
        results = model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
        assert np.isclose(m.coefficients[1], results.params.iloc[0],  atol=1e-2)

    def test_standard_errors(self, fe_model_two_way, panel_data):
        m = fe_model_two_way
        panel_data = panel_data.set_index(['id', 'year'])
        model = PanelOLS.from_formula('outcome ~ independent + TimeEffects + EntityEffects', data=panel_data)
        results = model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
        assert np.isclose(m.standard_errors[1], results.std_errors.iloc[0],  atol=1e-1)



#%%
