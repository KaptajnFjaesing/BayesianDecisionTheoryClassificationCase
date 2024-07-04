# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:43:05 2023

@author: 1056672
"""

#%%
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from scipy.special import gamma
import matplotlib.pyplot as plt
import os 

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

df_farm_training = pd.DataFrame(loaded_object).iloc[:600]
df_farm_test = pd.DataFrame(loaded_object).iloc[600:]


"""

The input data must have the structure (time, features), where time denotes
time dimension and features the feature dimension.

The target data must have dimensions (time, classes), where classes is the
number of classes.

"""
data_x_training = np.array(df_farm_training[["area","animals"]].values)
data_s_training = np.row_stack([(1, 0) if val else (0, 1) for val in df_farm_training["target"]])

data_x_test = np.array(df_farm_test[["area","animals"]].values)
data_s_test = np.row_stack([(1, 0) if val else (0, 1) for val in df_farm_test["target"]])
# %%

def setup(data_x:np.array, data_s: np.array) -> list:
    parameters = []
    parameters.append(np.random.rand(data_s.shape[1],data_x.shape[1]))# a
    parameters.append(np.random.rand(data_s.shape[1])) # b
    parameters.append(np.array([0.5,0.5])) # tau

    prior_parameters =[]
    for i in range(len(parameters[2])):
        prior_parameters.append(np.array([2.4,3]))

    ma = data_x.shape[0]/4+1
    mb = data_x.shape[0]/4+1
    mtau = 3

    mass = []
    mass.append(ma*np.ones(parameters[0].shape)) # masses for b1
    mass.append(mb*np.ones(parameters[1].shape)) # masses for W1
    mass.append(mtau*np.ones(parameters[2].shape))# masses for b2

    setup_data = []
    setup_data.append(data_x) 
    setup_data.append(data_s)
    setup_data.append(prior_parameters) 
    setup_data.append(parameters) 
    setup_data.append(mass)
    return setup_data

def momentum_initialization(mass: list) -> list:
    momentum =[]
    for i in range(len(mass)):
        momentum.append(np.random.normal(0, np.sqrt(mass[i]))) 
    return momentum
  
def hamiltonian_momentum_term(
        momentum: list,
        mass: list
        ) -> float:
    temp = []
    for i in range(len(momentum)):
        temp.append(np.sum(np.multiply(momentum[i],momentum[i])/mass[i]))
    return 1/2*np.sum(temp)

def class_conditional_probability(
        parameters: list,
        data_x: np.array
        ) -> np.array:
    T = np.exp(parameters[1]+np.einsum('kj,ij->ik', parameters[0], data_x))
    return T / T.sum(axis=1, keepdims=True)

def hamiltonian(
        data_s: np.array,
        data_x: np.array,
        parameters: list,
        hamiltonian_momentum_term: float,
        prior_parameters: list
        ) -> float: 
    f = class_conditional_probability(parameters = parameters, data_x = data_x)
    a = parameters[0]
    b = parameters[1]
    tau_a = parameters[2][0]
    tau_b = parameters[2][1]
    
    alpha_tau_a = prior_parameters[0][0]
    beta_tau_a = prior_parameters[0][1]
    alpha_tau_b = prior_parameters[1][0]
    beta_tau_b = prior_parameters[1][1]
    
    h1 = []
    h1.append(hamiltonian_momentum_term) # Momentum term
    h1.append(-np.sum(data_s*np.log(f)) ) # Loss term
    h1.append(np.log(gamma(alpha_tau_a))-alpha_tau_a*np.log(beta_tau_a)-alpha_tau_a*tau_a+beta_tau_a*np.exp(tau_a))
    h1.append(np.log(gamma(alpha_tau_b))-alpha_tau_b*np.log(beta_tau_b)-alpha_tau_b*tau_b+beta_tau_b*np.exp(tau_b))
    h1.append((1/2)*(np.log(2*np.pi)-tau_a)+(np.exp(tau_a)/2)*np.sum(a**2))
    h1.append((1/2)*(np.log(2*np.pi)-tau_b)+(np.exp(tau_b)/2)*np.sum(b**2))
    return sum(h1)

def gradient(
        data_s: np.array,
        data_x: np.array,
        parameters: list,
        prior_parameters: list
        ) -> list:
    f = class_conditional_probability(parameters = parameters, data_x = data_x)
    a = parameters[0]
    b = parameters[1]
    tau_a = parameters[2][0]
    tau_b = parameters[2][1]

    alpha_tau_a = prior_parameters[0][0]
    beta_tau_a = prior_parameters[0][1]
    alpha_tau_b = prior_parameters[1][0]
    beta_tau_b = prior_parameters[1][1]

    gradient_a = np.einsum('il,ik->kl', data_x, f-data_s)+np.exp(tau_a)*a
    gradient_b = np.einsum('ik->k', f-data_s)+np.exp(tau_b)*b
    gradient_tau_a = -alpha_tau_a+beta_tau_a*np.exp(tau_a)-1/2+(np.exp(tau_a)/2)*np.sum(a**2)
    gradient_tau_b = -alpha_tau_b+beta_tau_b*np.exp(tau_b)-1/2+(np.exp(tau_b)/2)*np.sum(b**2)
    
    gradient_data = []
    gradient_data.append(gradient_a)
    gradient_data.append(gradient_b)
    gradient_data.append(np.array([gradient_tau_a, gradient_tau_b]))
    return gradient_data

def HMC(
        markov_chain_length: int,
        step_scale: int,
        number_of_steps_scale: int,
        setup_data: list,
        ) -> tuple:
    # Unpack setup data
    data_x = setup_data[0]
    data_s = setup_data[1]
    prior_parameters = setup_data[2]
    parameters = setup_data[3]
    mass = setup_data[4]
    
    # Define output variable structures
    potential_values = np.zeros(markov_chain_length)
    hamiltonian_values = np.zeros(markov_chain_length)
    parameter_samples = []
    transition_probabilities = np.zeros(markov_chain_length)
    
    for k in tqdm (range (markov_chain_length), desc="Progress"):
        eps = np.divide(1+np.random.rand(),step_scale)
        momentum = momentum_initialization(mass)
        hamiltonian_momentum_term_value = hamiltonian_momentum_term(
            momentum = momentum,
            mass = mass
            )
        initial_hamiltonian = hamiltonian(
                data_s = data_s,
                data_x = data_x,
                parameters = parameters,
                hamiltonian_momentum_term = hamiltonian_momentum_term_value,
                prior_parameters = prior_parameters
                )
        initial_potential = initial_hamiltonian-hamiltonian_momentum_term_value
        initial_parameters = parameters[:]
        
        # Half a momentum step
        numerical_gradient = gradient(
                data_s = data_s,
                data_x = data_x,
                parameters = parameters,
                prior_parameters = prior_parameters
                )
        for i in range(len(parameters)):
            momentum[i] = momentum[i]-(eps/2)*numerical_gradient[i]
        # Initiate dive
        number_of_steps = 2*random.randint(int(number_of_steps_scale/2),int((number_of_steps_scale+10)/2))
        for j in range(number_of_steps):
            for i in range(len(parameters)):
                parameters[i] = parameters[i]+eps*momentum[i]/mass[i]
            if j<number_of_steps-1:
                numerical_gradient = gradient(
                        data_s = data_s,
                        data_x = data_x,
                        parameters = parameters,
                        prior_parameters = prior_parameters
                        )
                for i in range(len(parameters)):
                    momentum[i] = momentum[i]-eps*numerical_gradient[i]
        numerical_gradient = gradient(
                data_s = data_s,
                data_x = data_x,
                parameters = parameters,
                prior_parameters = prior_parameters
                )
        for i in range(len(parameters)):
            momentum[i] = momentum[i]-(eps/2)*numerical_gradient[i]
        
        hamiltonian_momentum_term_value = hamiltonian_momentum_term(
            momentum = momentum,
            mass = mass
            )
        final_hamiltonian = hamiltonian(
                data_s = data_s,
                data_x = data_x,
                parameters = parameters,
                hamiltonian_momentum_term = hamiltonian_momentum_term_value,
                prior_parameters = prior_parameters
                )
        final_potential = final_hamiltonian-hamiltonian_momentum_term_value
        u = np.random.rand()
        Pt = min(1,np.exp(initial_hamiltonian-final_hamiltonian))
        if Pt == 1 or u < Pt:
            potential_values[k] = final_potential
            hamiltonian_values[k] = final_hamiltonian
        else:
            parameters = initial_parameters[:]
            potential_values[k] = initial_potential
            hamiltonian_values[k] = initial_hamiltonian
        parameter_samples.append(parameters[:])
        transition_probabilities[k] = np.exp(initial_hamiltonian-final_hamiltonian)
    return (
        potential_values,
        hamiltonian_values,
        parameter_samples,
        transition_probabilities
        )

def HMCtrain(
        markov_chain_length: int,
        step_scale: int,
        number_of_steps_scale : int,
        data_x: np.array,
        data_s: np.array
        ) -> tuple:
    return HMC(
        markov_chain_length = markov_chain_length,
        step_scale = step_scale,
        number_of_steps_scale = number_of_steps_scale,
        setup_data = setup(
            data_x = data_x,
            data_s = data_s
            )
        )


# %% Result plot

step_scale = 50
number_of_steps_scale = 100


HMC_result = HMCtrain(
        markov_chain_length = 2000,
        step_scale = step_scale,
        number_of_steps_scale = number_of_steps_scale,
        data_x = data_x_training,
        data_s = data_s_training
        )

# %% Plot Hamiltonian and potential
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


# %% Use parameters to compute a prediction (S = s_1 is predicted_class_probabilities[0])

sampled_parameters = HMC_result[2][500:]

predicted_class_probabilities = np.mean([class_conditional_probability(
    parameters = sampled_parameter,
    data_x = data_x_training
    ) for sampled_parameter in sampled_parameters], axis = 0)

targets = [1 if a>b else 0 for a,b in predicted_class_probabilities]
true_targets = [1 if a>b else 0 for a,b in data_s_training]
training_accuracy = [1 if a == b else 0 for a,b, in list(zip(targets,true_targets))]
print(sum(training_accuracy)/len(training_accuracy))

results = []
for i in range(len(training_accuracy)):
    if training_accuracy[i] == 0:
        results.append([predicted_class_probabilities[i],data_x_training[i]])


# %% Use parameters to compute a prediction (S = s_1 is predicted_class_probabilities[0])

predicted_class_probabilities = np.mean([class_conditional_probability(
    parameters = sampled_parameter,
    data_x = data_x_test
    ) for sampled_parameter in sampled_parameters], axis = 0)

targets = [1 if a>b else 0 for a,b in predicted_class_probabilities]
true_targets = [1 if a>b else 0 for a,b in data_s_test]
training_accuracy = [1 if a == b else 0 for a,b, in list(zip(targets,true_targets))]
print(sum(training_accuracy)/len(training_accuracy))
results = []
for i in range(len(training_accuracy)):
    if training_accuracy[i] == 0:
        results.append([predicted_class_probabilities[i],data_x_test[i]])

#%% Function tests

setup_data = setup(data_x = data_x_training, data_s = data_s_training)

data_x = setup_data[0]
data_s = setup_data[1]
prior_parameters = setup_data[2]
parameters = setup_data[3]
mass = setup_data[4]

f_test = class_conditional_probability(
        parameters = parameters,
        data_x = data_x
        )

momentum1 = momentum_initialization(mass = mass)

hest = hamiltonian_momentum_term(
    momentum = momentum1,
    mass = mass
    )

hamiltonian1 = hamiltonian(
        data_s = data_s,
        data_x = data_x,
        parameters = parameters,
        momentum = momentum1,
        mass = mass,
        prior_parameters = prior_parameters
        )

gradient_data = gradient(
    data_s = data_s,
    data_x = data_x,
    parameters = parameters,
    momentum = momentum1,
    mass = mass,
    prior_parameters = prior_parameters
    )



# %% Gradient test
import copy

epsilon = 0.000001
parameters_perturbed = copy.deepcopy(parameters)
parameters_perturbed[0][0][0] = parameters[0][0][0]+epsilon

H1 = hamiltonian(
        data_s = data_s,
        data_x = data_x,
        parameters = parameters,
        momentum = momentum1,
        mass = mass,
        prior_parameters = prior_parameters
        )

H2 = hamiltonian(
        data_s = data_s,
        data_x = data_x,
        parameters = parameters_perturbed,
        momentum = momentum1,
        mass = mass,
        prior_parameters = prior_parameters
        )

numerical_gradient = (H2-H1)/epsilon

print("Numerical Gradient :", numerical_gradient)
print("Analytical Gradient :", gradient_data[0])
