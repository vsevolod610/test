# -*- coding: utf-8 -*-
"""
MCMC analyze example "Dots"
"""

import numpy as np

from core_mcmc.mcmc_quick import mcmc_quick


np.random.seed(123)


def line(a, b, x):
    return a * x + b


# model
ndim = 2
model = line
params_names = ['a', 'b']


# data
N = 100
a_true, b_true = 1, 2
x_min, x_max = -10, 20
mu, sigma = 0, 1

x = x_min + (x_max - x_min) * np.random.rand(N)
y = model(a_true, b_true, x) + np.random.normal(mu, sigma, N)


# settings
params_try = [0.9, 1.2]
params_sigma = [0.5, 1]
init = np.array([params_try, params_sigma]).T

nwalkers = 400
nsteps = 300
amputate = int(0.3 * nsteps)


# mcmc 
data_params = {
        'x' : x, 
        'y' : y
        }
model_params = {
        'model' : model, 
        'init' : init,
        }
settings_params = {
        'nwalkers': nwalkers, 
        'nsteps' : nsteps, 
        'amputate' : amputate
        }

mcmc_quick(data_params, model_params, settings_params,
           prnt=True, show=True, save=False)

