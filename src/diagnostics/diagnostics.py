import numpy as np
import pandas as pd
import scipy.stats as stats

from src.models.linear_model import LinearModel
from src.models.panel import FixedEffects


def f_test(model1, model2):
    """
    Perform an F-test to compare two models.

    Args:
    model1: The model.
    model2: The model to compare with model1.

    Returns:
    f_stat: The F-statistic for the test.
    dfn: Degrees of freedom numerator.
    dfd: Degrees of freedom denominator.
    p_value: The p-value for the F-test.
    """
    rss_restricted = model1.rss
    rss_unrestricted = model2.rss
    if isinstance(model1, FixedEffects):
        model_1_regs = model1.n_regs + len(model1.dummy_cols)
    else:
        model_1_regs = model1.n_regs

    if isinstance(model2, FixedEffects):
        model_2_regs = model2.n_regs + len(model2.dummy_cols)
    else:
        model_2_regs = model2.n_regs

    dfn = model_2_regs - model_1_regs
    dfd = model1.obs - model_2_regs - 1

    f_stat = ((rss_restricted - rss_unrestricted) / dfn) / (rss_unrestricted / dfd)
    p_value = 1 - stats.f.cdf(f_stat, dfn, dfd)

    results = pd.DataFrame({
        'F-Statistic': [f_stat],
        'DFN': [dfn],
        'DFD': [dfd],
        'p-value': [p_value]
    })
    print("F-Test Results for significance of add effects")
    print("==============================================")
    print("Alternative Hypothesis: Added covariates are significant")
    return results

#%%
