# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:17:07 2024

@author: 1056672
"""
# %%
import pandas as pd
import numpy as np
import pymc as pm
import pickle
import HMC as hmc
import matplotlib.pyplot as plt

# Specify the path to your pickle file
pickle_file_path = r'..\data\mock_data.pkl'

# Check if the file exists
if not os.path.exists(pickle_file_path):
    print(f"File not found: {pickle_file_path}")
else:
    # Open the file in binary mode
    with open(pickle_file_path, 'rb') as file:
        # Load the object from the file
        loaded_object = pickle.load(file)
    print("File loaded successfully")


def prediction_accuracy_hmc(HMC_result, burn_in, data_s, data_x):

    sampled_parameters = HMC_result[2][burn_in:]
    
    predicted_class_probabilities = np.mean([hmc.class_conditional_probability(
        parameters = sampled_parameter,
        data_x = data_x
        ) for sampled_parameter in sampled_parameters], axis = 0)
    
    targets = [1 if a>b else 0 for a,b in predicted_class_probabilities]
    true_targets = [1 if a>b else 0 for a,b in data_s]
    accuracy = [1 if a == b else 0 for a,b, in list(zip(targets,true_targets))]
    
    results = []
    for i in range(len(accuracy)):
        if accuracy[i] == 0:
            results.append([predicted_class_probabilities[i],data_x[i]])
    return accuracy, results

def construct_pymc_model(data_x_training,data_s_training):
    with pm.Model() as model:
        n_features = data_x_training.shape[1]
        n_classes = data_s_training.shape[1]
        x = pm.MutableData("x", data_x_training)
        y_obs = data_s_training
    
        precision_a = pm.Gamma('precision_a', alpha=2.4, beta=3)
        precision_b = pm.Gamma('precision_b', alpha=2.4, beta=3)
        param_a = pm.Normal('param_a', 0, sigma=1/np.sqrt(precision_a), shape=(n_classes, n_features))
        param_b = pm.Normal('param_b', 0, sigma=1/np.sqrt(precision_b), shape=(n_classes,))
        T = pm.Deterministic('T', pm.math.exp(param_b + pm.math.dot(x, param_a.T)))
        class_conditional_probability = pm.Deterministic('class_conditional_probability', T / T.sum(axis=1, keepdims=True))
        pm.Multinomial('y', n=1, p=class_conditional_probability,shape=x.shape, observed=y_obs)
        return model
    
def prediction_accuracy_pymc(model, posterior, data_s, data_x):

    with model:
        pm.set_data({"x": data_x})
        posterior_predictive = pm.sample_posterior_predictive(posterior, predictions=True)
    
    predicted_class_probabilities = posterior_predictive.predictions['y'].mean((('draw','chain'))).values

    targets = [1 if a>b else 0 for a,b in predicted_class_probabilities]
    true_targets = [1 if a>b else 0 for a,b in data_s]
    accuracy = [1 if a == b else 0 for a,b, in list(zip(targets,true_targets))]
    
    results = []
    for i in range(len(accuracy)):
        if accuracy[i] == 0:
            results.append([predicted_class_probabilities[i],data_x[i]])
    return accuracy, results
 
# Split the data into training and test sets
df_farm_training = pd.DataFrame(loaded_object).iloc[:600]
df_farm_test = pd.DataFrame(loaded_object).iloc[600:]

# Prepare the data
data_x_training = np.array(df_farm_training[["area","animals"]].values)
data_s_training = np.row_stack([(1, 0) if val else (0, 1) for val in df_farm_training["target"]])

data_x_test = np.array(df_farm_test[["area","animals"]].values)
data_s_test = np.row_stack([(1, 0) if val else (0, 1) for val in df_farm_test["target"]])


# %% HMC

step_scale = 20
number_of_steps_scale = 1000

HMC_result = hmc.HMCtrain(
        markov_chain_length = 2000,
        step_scale = step_scale,
        number_of_steps_scale = number_of_steps_scale,
        data_x = data_x_training,
        data_s = data_s_training
        )

#%%  Plot Hamiltonian and potential
plt.figure(2,figsize=(10,7))
plt.title(f"Step scale {step_scale}")
plt.plot(HMC_result[0],label='Potential',linewidth=1.0)
plt.plot(HMC_result[1],label='Hamiltonian',linewidth=1.0)
plt.ylabel("Energy", fontsize=18)
plt.xlabel("Iteration", fontsize=18)
plt.xticks( fontsize=18)
plt.yticks( fontsize=18)
plt.minorticks_on()
plt.legend(fontsize=18)

# %%
acuracy_training_hmc, results_training_hmc = prediction_accuracy_hmc(HMC_result = HMC_result, burn_in = 900, data_s = data_s_training, data_x = data_x_training)
acuracy_test_hmc, results_test_hmc = prediction_accuracy_hmc(HMC_result = HMC_result, burn_in = 900, data_s = data_s_test, data_x = data_x_test)


print(results_test_hmc)
# %% pymc

model = construct_pymc_model(data_x_training,data_s_training)
with model:
    posterior = pm.sample(tune=500, draws=1500, chains=1)


# %%
acuracy_training_pymc, results_training_pymc = prediction_accuracy_pymc(model = model, posterior = posterior, data_s = data_s_training, data_x = data_x_training)
acuracy_test_pymc, results_test_pymc = prediction_accuracy_pymc(model = model, posterior = posterior, data_s = data_s_test, data_x = data_x_test)




