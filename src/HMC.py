# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:59:48 2024

@author: 1056672
"""

import numpy as np
import random
from tqdm import tqdm
from scipy.special import gamma
from scipy.special import softmax

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


def class_conditional_probability(parameters: list, data_x: np.array) -> np.array:
    # Compute the scores
    scores = parameters[1] + np.einsum('kj,ij->ik', parameters[0], data_x)
    
    # Compute the probabilities using softmax
    probabilities = softmax(scores, axis=1)
    return probabilities

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