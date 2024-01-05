import numpy as np
import pandas as pd
import statsmodels.api as sm
import pytest

from src.models.linreg import LinReg


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
    np.random.seed(69)
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
    np.random.seed(69)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.normal(0, x, 100)
    data = pd.DataFrame({'outcome': y, 'independent': x})
    return data


@pytest.fixture(scope="class")
def linreg_model_robust(regression_data_robust):
    model = LinReg(df=regression_data_robust,
                   outcome="outcome",
                   independent=["independent"],
                   standard_error_type='hc0')
    return model


@pytest.fixture(scope="class")
def sm_model_robust(regression_data_robust):
    x = regression_data_robust["independent"]
    y = regression_data_robust["outcome"]
    x = sm.add_constant(x)
    return sm.OLS(y, x).fit(cov_type='HC0')


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
    np.random.seed(69)
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


@pytest.fixture(scope="class")
def regression_data_all_operator_slr():
    np.random.seed(69)
    x = np.linspace(0, 10, 100)
    y = 5 * x + 10 + np.random.normal(0, 2, 100)
    data = pd.DataFrame({'outcome': y, 'independent': x})
    return data


@pytest.fixture(scope="class")
def linreg_model_all_operator_slr(regression_data_all_operator_slr):
    return LinReg(df=regression_data_all_operator_slr, outcome="outcome", independent=["."])


@pytest.fixture(scope="class")
def sm_model_all_operator_slr(regression_data_all_operator_slr):
    x = regression_data_all_operator_slr["independent"]
    y = regression_data_all_operator_slr["outcome"]
    x = sm.add_constant(x)
    return sm.OLS(y, x).fit()


class TestRegressionAllOperatorSlr:

    def test_coefficients(self, linreg_model_all_operator_slr, sm_model_all_operator_slr):
        assert np.isclose(linreg_model_all_operator_slr.coefficients[0],
                          sm_model_all_operator_slr.params.iloc[0],
                          atol=1e-2)
        assert np.isclose(linreg_model_all_operator_slr.coefficients[1],
                          sm_model_all_operator_slr.params.iloc[1],
                          atol=1e-2)

    def test_standard_errors(self, linreg_model_all_operator_slr, sm_model_all_operator_slr):
        assert np.isclose(linreg_model_all_operator_slr.standard_errors[0],
                          sm_model_all_operator_slr.bse.iloc[0],
                          atol=1e-2)
        assert np.isclose(linreg_model_all_operator_slr.standard_errors[1],
                          sm_model_all_operator_slr.bse.iloc[1],
                          atol=1e-2)

    def test_p_values(self, linreg_model_all_operator_slr, sm_model_all_operator_slr):
        assert np.isclose(linreg_model_all_operator_slr.p_values[0],
                          sm_model_all_operator_slr.pvalues.iloc[0],
                          atol=1e-2)
        assert np.isclose(linreg_model_all_operator_slr.p_values[1],
                          sm_model_all_operator_slr.pvalues.iloc[1],
                          atol=1e-2)

    def test_t_statistics(self, linreg_model_all_operator_slr, sm_model_all_operator_slr):
        assert np.isclose(linreg_model_all_operator_slr.t_stats[0],
                          sm_model_all_operator_slr.tvalues.iloc[0],
                          atol=1e-2)
        assert np.isclose(linreg_model_all_operator_slr.t_stats[1],
                          sm_model_all_operator_slr.tvalues.iloc[1],
                          atol=1e-2)

    def test_confint(self, linreg_model_all_operator_slr, sm_model_all_operator_slr):
        assert np.isclose(linreg_model_all_operator_slr.conf_int[0][0],
                          sm_model_all_operator_slr.conf_int().iloc[0, 0],
                          atol=1e-2)
        assert np.isclose(linreg_model_all_operator_slr.conf_int[0][1],
                          sm_model_all_operator_slr.conf_int().iloc[0, 1],
                          atol=1e-2)
        assert np.isclose(linreg_model_all_operator_slr.conf_int[1][0],
                          sm_model_all_operator_slr.conf_int().iloc[1, 0],
                          atol=1e-2)
        assert np.isclose(linreg_model_all_operator_slr.conf_int[1][1],
                          sm_model_all_operator_slr.conf_int().iloc[1, 1],
                          atol=1e-2)


@pytest.fixture(scope="class")
def regression_data_all_operator_mlr():
    np.random.seed(69)
    x = np.linspace(0, 5, 50)
    z = np.linspace(3, 8, 50)
    y = 1 + 3 * x + 5*z + np.random.normal(0, 2, 50)
    data = pd.DataFrame({'outcome': y,
                         'independent1': x,
                         'independent2': z})
    return data


@pytest.fixture(scope="class")
def linreg_model_all_operator_mlr(regression_data_all_operator_mlr):
    return LinReg(df=regression_data_all_operator_mlr, outcome="outcome", independent=["."])


@pytest.fixture(scope="class")
def sm_model_all_operator_mlr(regression_data_all_operator_mlr):
    x = regression_data_all_operator_mlr[['independent1', 'independent2']]
    y = regression_data_all_operator_mlr['outcome']
    x = sm.add_constant(x)
    return sm.OLS(y, x).fit()


class TestRegressionAllOperatorMlr:

    def test_coefficients(self, linreg_model_all_operator_mlr, sm_model_all_operator_mlr):
        assert np.isclose(linreg_model_all_operator_mlr.coefficients[0],
                          sm_model_all_operator_mlr.params.iloc[0],
                          atol=1e-1)
        assert np.isclose(linreg_model_all_operator_mlr.coefficients[1],
                          sm_model_all_operator_mlr.params.iloc[1],
                          atol=1e-1)

    def test_standard_errors(self, linreg_model_all_operator_mlr, sm_model_all_operator_mlr):
        assert np.isclose(linreg_model_all_operator_mlr.standard_errors[0],
                          sm_model_all_operator_mlr.bse.iloc[0],
                          atol=1e-1)
        assert np.isclose(linreg_model_all_operator_mlr.standard_errors[1],
                          sm_model_all_operator_mlr.bse.iloc[1],
                          atol=1e-1)

    def test_p_values(self, linreg_model_all_operator_mlr, sm_model_all_operator_mlr):
        assert np.isclose(linreg_model_all_operator_mlr.p_values[0],
                          sm_model_all_operator_mlr.pvalues.iloc[0],
                          atol=1e-1)
        assert np.isclose(linreg_model_all_operator_mlr.p_values[1],
                          sm_model_all_operator_mlr.pvalues.iloc[1],
                          atol=1e-1)

    def test_t_statistics(self, linreg_model_all_operator_mlr, sm_model_all_operator_mlr):
        assert np.isclose(linreg_model_all_operator_mlr.t_stats[0],
                          sm_model_all_operator_mlr.tvalues.iloc[0],
                          atol=1e-1)
        assert np.isclose(linreg_model_all_operator_mlr.t_stats[1],
                          sm_model_all_operator_mlr.tvalues.iloc[1],
                          atol=1e-1)

    def test_confint(self, linreg_model_all_operator_mlr, sm_model_all_operator_mlr):
        assert np.isclose(linreg_model_all_operator_mlr.conf_int[0][0],
                          sm_model_all_operator_mlr.conf_int().iloc[0, 0],
                          atol=1e-1)
        assert np.isclose(linreg_model_all_operator_mlr.conf_int[0][1],
                          sm_model_all_operator_mlr.conf_int().iloc[0, 1],
                          atol=1e-1)
        assert np.isclose(linreg_model_all_operator_mlr.conf_int[1][0],
                          sm_model_all_operator_mlr.conf_int().iloc[1, 0],
                          atol=1e-1)
        assert np.isclose(linreg_model_all_operator_mlr.conf_int[1][1],
                          sm_model_all_operator_mlr.conf_int().iloc[1, 1],
                          atol=1e-1)


@pytest.fixture(scope="class")
def regression_data_not_operator_mlr():
    np.random.seed(69)
    x = np.linspace(0, 5, 50)
    z = np.linspace(3, 8, 50)
    y = 1 + 3 * x + 5*z + np.random.normal(0, 2, 50)
    data = pd.DataFrame({'outcome': y,
                         'independent1': x,
                         'independent2': z})
    return data


@pytest.fixture(scope="class")
def linreg_model_not_operator_mlr(regression_data_not_operator_mlr):
    return LinReg(df=regression_data_not_operator_mlr, outcome="outcome", independent=["!independent2"])


@pytest.fixture(scope="class")
def sm_model_not_operator_mlr(regression_data_not_operator_mlr):
    x = regression_data_not_operator_mlr['independent1']
    y = regression_data_not_operator_mlr['outcome']
    x = sm.add_constant(x)
    return sm.OLS(y, x).fit()


class TestRegressionNotOperatorMlr:

    def test_coefficients(self, linreg_model_not_operator_mlr, sm_model_not_operator_mlr):
        assert np.isclose(linreg_model_not_operator_mlr.coefficients[0],
                          sm_model_not_operator_mlr.params.iloc[0],
                          atol=1e-1)
        assert np.isclose(linreg_model_not_operator_mlr.coefficients[1],
                          sm_model_not_operator_mlr.params.iloc[1],
                          atol=1e-1)

    def test_standard_errors(self, linreg_model_not_operator_mlr, sm_model_not_operator_mlr):
        assert np.isclose(linreg_model_not_operator_mlr.standard_errors[0],
                          sm_model_not_operator_mlr.bse.iloc[0],
                          atol=1e-1)
        assert np.isclose(linreg_model_not_operator_mlr.standard_errors[1],
                          sm_model_not_operator_mlr.bse.iloc[1],
                          atol=1e-1)


@pytest.fixture(scope="class")
def regression_data_exclude_multiple():
    np.random.seed(69)
    x1, x2, x3 = np.random.randn(50), np.random.randn(50), np.random.randn(50)
    y = 2 * x1 + 3 * x2 + 4 * x3 + np.random.randn(50)
    data = pd.DataFrame({'outcome': y,
                         'independent1': x1,
                         'independent2': x2,
                         'independent3': x3})
    return data


@pytest.fixture(scope="class")
def linreg_model_exclude_multiple(regression_data_exclude_multiple):
    return LinReg(df=regression_data_exclude_multiple,
                  outcome="outcome",
                  independent=["!independent2",
                               "!independent3"])


@pytest.fixture(scope="class")
def sm_model_exclude_multiple(regression_data_exclude_multiple):
    x = regression_data_exclude_multiple[['independent1']]
    y = regression_data_exclude_multiple['outcome']
    x = sm.add_constant(x)
    return sm.OLS(y, x).fit()


class TestRegressionExcludeMultiple:

    def test_coefficients(self, linreg_model_exclude_multiple, sm_model_exclude_multiple):
        np.testing.assert_allclose(linreg_model_exclude_multiple.coefficients,
                                   sm_model_exclude_multiple.params,
                                   atol=1e-1)

    def test_standard_errors(self, linreg_model_exclude_multiple, sm_model_exclude_multiple):
        np.testing.assert_allclose(linreg_model_exclude_multiple.standard_errors,
                                   sm_model_exclude_multiple.bse,
                                   atol=1e-1)


@pytest.fixture(scope="class")
def linreg_model_no_exclusion(regression_data_exclude_multiple):
    return LinReg(df=regression_data_exclude_multiple,
                  outcome="outcome",
                  independent=["independent1",
                               "independent2",
                               "independent3"])


@pytest.fixture(scope="class")
def sm_model_no_exclusion(regression_data_exclude_multiple):
    x = regression_data_exclude_multiple[['independent1',
                                          'independent2',
                                          'independent3']]
    y = regression_data_exclude_multiple['outcome']
    x = sm.add_constant(x)
    return sm.OLS(y, x).fit()


class TestRegressionNoExclusion:

    def test_coefficients(self, linreg_model_no_exclusion, sm_model_no_exclusion):
        np.testing.assert_allclose(linreg_model_no_exclusion.coefficients,
                                   sm_model_no_exclusion.params,
                                   atol=1e-1)

    def test_standard_errors(self, linreg_model_no_exclusion, sm_model_no_exclusion):
        np.testing.assert_allclose(linreg_model_no_exclusion.standard_errors,
                                   sm_model_no_exclusion.bse,
                                   atol=1e-1)


@pytest.fixture(scope="class")
def linreg_model_exclude_all(regression_data_exclude_multiple):
    return LinReg(df=regression_data_exclude_multiple,
                  outcome="outcome",
                  independent=["!independent1",
                               "!independent2",
                               "!independent3"])


class TestRegressionExcludeAll:

    def test_linreg_model_exclude_nonexistent_raises_error(self, regression_data_exclude_multiple):
        with pytest.raises(ValueError) as excinfo:
            LinReg(df=regression_data_exclude_multiple,
                   outcome="outcome",
                   independent=["!independent1", "!independent2", "!independent3"])
        assert 'Oops, you cant exclude all columns from your data.' in str(excinfo.value)


@pytest.fixture(scope="class")
def linreg_model_exclude_nonexistent(regression_data_exclude_multiple):
    return LinReg(df=regression_data_exclude_multiple,
                  outcome="outcome",
                  independent=["!nonexistent",
                               "independent1",
                               "independent2",
                               "independent3"])


class TestRegressionExcludeNonexistent:

    def test_linreg_model_exclude_nonexistent_raises_error(self, regression_data_exclude_multiple):
        with pytest.raises(ValueError) as excinfo:
            LinReg(df=regression_data_exclude_multiple,
                   outcome="outcome",
                   independent=["!nonexistent", "independent1", "independent2", "independent3"])
        assert "Oops, nonexistent is not a column in your data. Check if youve made a typo." in str(excinfo.value)


@pytest.fixture(scope="class")
def data_with_base_variable():
    """DataFrame with a base variable already present"""
    np.random.seed(69)
    x = np.random.randn(50)
    y = 2 * x + np.random.randn(50)
    data = pd.DataFrame({'x': x, 'y': y})
    return data


@pytest.fixture(scope="class")
def data_without_base_variable():
    """DataFrame without the base variable present"""
    np.random.seed(69)
    y = np.random.randn(50)
    data = pd.DataFrame({'y': y})
    return data


@pytest.fixture(scope="class")
def model_with_transformation(data_with_base_variable):
    return LinReg(df=data_with_base_variable,
                  outcome="y",
                  independent=["x^3"])


@pytest.fixture(scope="class")
def model_with_transformation_missing_base(data_without_base_variable):
    return LinReg(df=data_without_base_variable,
                  outcome="y",
                  independent=["x^3"])


class TestModelWithTransformation:
    def test_independent_vars_include_transformed(self, model_with_transformation):
        assert "x^1" in model_with_transformation.independent_vars
        assert "x^2" in model_with_transformation.independent_vars
        assert "x^3" in model_with_transformation.independent_vars

    def test_data_includes_transformed_columns(self, model_with_transformation):
        assert "x^1" in model_with_transformation.data.columns
        assert "x^2" in model_with_transformation.data.columns
        assert "x^3" in model_with_transformation.data.columns


class TestModelWithTransformationMissingBase:

    def test_independent_vars_include_transformed(self, data_without_base_variable):
        with pytest.raises(ValueError) as excinfo:
            LinReg(df=data_without_base_variable,
                   outcome="y",
                   independent=["x^3"])
        assert f"Base variable 'x' not found in DataFrame. Check for a typo in 'x^3'." in str(excinfo.value)


class TestModelWithTransformationBadChar:

    def test_independent_vars_include_transformed(self, data_without_base_variable):
        with pytest.raises(ValueError) as excinfo:
            LinReg(df=data_without_base_variable,
                   outcome="y",
                   independent=["x^i"])
        assert f"Invalid exponent 'i' in variable 'x^i'. Check for a typo." in str(excinfo.value)


@pytest.fixture(scope="class")
def data_with_variables():
    """DataFrame with required variables for interaction"""
    np.random.seed(69)
    x = np.random.randn(50)
    z = np.random.randn(50)
    y = 2 * x + 3 * z + np.random.randn(50)
    data = pd.DataFrame({'x': x, 'z': z, 'y': y})
    return data


@pytest.fixture(scope="class")
def data_missing_variables():
    """DataFrame missing one or more required variables for interaction"""
    np.random.seed(69)
    z = np.random.randn(50)
    y = 3 * z + np.random.randn(50)
    data = pd.DataFrame({'z': z, 'y': y})
    return data


class TestBasicInteractionOperator:

    def test_interaction_created(self, data_with_variables):
        model = LinReg(df=data_with_variables,
                       outcome="y",
                       independent=["x", "z", "x:z"])

        assert 'x:z' in model.data.columns
        assert 'x:z' in model.independent_vars

    def test_raises_error_missing_variable(self, data_missing_variables):

        with pytest.raises(ValueError) as excinfo:
            LinReg(df=data_missing_variables,
                   outcome="y",
                   independent=["x", "z", "x:z"])

        assert "Variable 'x' not found in DataFrame." in str(excinfo.value)


class TestAdvancedInteractionOperator:

    def test_interaction_and_individual_vars_created(self, data_with_variables):
        model = LinReg(df=data_with_variables,
                       outcome="y",
                       independent=["x", "z", "x*z"])

        assert 'x' in model.data.columns and 'z' in model.data.columns
        assert 'x*z' in model.data.columns
        assert 'x*z' in model.independent_vars
        assert 'x' in model.independent_vars and 'z' in model.independent_vars

    def test_raises_error_missing_variable(self, data_missing_variables):

        with pytest.raises(ValueError) as excinfo:
            LinReg(df=data_missing_variables,
                   outcome="y",
                   independent=["x", "z", "x*z"])
        assert "Variable 'x' not found in DataFrame." in str(excinfo.value)





