from src.models import nonparametric as npm
import numpy as np
import pandas as pd

control_values = np.random.normal(0, 3, 1000)
control_treatment = np.zeros(1000)

control = pd.DataFrame()

control = control.assign(values=control_values, treatment=control_treatment)

treatment_values = np.random.normal(3, 3, 1000)
treatment_treatment = np.ones(1000)

treatment = pd.DataFrame()

treatment = treatment.assign(values=treatment_values, treatment=treatment_treatment)

test_data = pd.concat([treatment, control], axis=0)


def test_fit():

    model = npm.PermInf(df=test_data,
                        treatment_column='treatment',
                        outcome_column='values',
                        simulations=3)

    model.fit()


def test_power():
    #power = npm.Power().fit()

    power2= npm.Power(control_mean=10,treatment_mean= 11, control_sd= 3, treatment_sd=3.5).fit()