# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:43:05 2023

@author: 1056672
"""

import pickle
import pandas as pd
# Specify the path to your pickle file
pickle_file_path = r'C:\Users\1056672\OneDrive - VELUX\Documents\workspace_research_scientist\commit_projects\farm\src\mock_data.pkl'

# Open the file in binary mode
with open(pickle_file_path, 'rb') as file:
    # Load the object from the file
    loaded_object = pickle.load(file)

df_farm = pd.DataFrame(loaded_object)


# %%
ko = df_farm[df_farm["id"] == "5zEogVTG"]


import pymc3 as pm
import numpy as np
import pandas as pd

# Generate some example data
np.random.seed(42)
size = 100
X = np.random.randn(size, 1)
true_intercept = 1
true_slope = 2
true_probs = 1 / (1 + np.exp(-(true_intercept + true_slope * X.flatten())))
y = np.random.binomial(1, true_probs)

# Create a PyMC3 model
with pm.Model() as logistic_model:
    # Define priors for the parameters
    intercept = pm.Normal('intercept', mu=0, sd=10)
    slope = pm.Normal('slope', mu=0, sd=10)

    # Define the likelihood function
    p = pm.math.sigmoid(intercept + slope * X.flatten())
    y_obs = pm.Bernoulli('y_obs', p=p, observed=y)

# Perform inference
with logistic_model:
    trace = pm.sample(2000, tune=1000)

# Plot the posterior distributions
pm.traceplot(trace)
