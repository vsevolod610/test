# -*- coding: utf-8 -*-
"""
MCMC analyze example "Sin"
"""

import numpy as np

from core_mcmc.mcmc_quick import mcmc_quick


np.random.seed(123)


def sin_model(a, b, c, x0, z, x):
    return a * np.sin(c * (x - x0) ** z) + b 


# model
params_names = ['a', 'b', 'c', 'x0']
const_names = ['z']
model = sin_model
ndim = len(params_names)

prior_data = dict()
prior_data['box'] = {params_names.index('a'): [2, 8],
                     params_names.index('b'): [1, 4]}
prior_data['gauss'] = {params_names.index('b'): [2, 0.1, 0.1]}
prior_data['const'] = {const_names.index('z'): 1.0}

# FIXME: order of values not gгuarantee in python < 3.7 
const = list(prior_data['const'].values())


# data
params_true = [5, 2, 1, 0]
N = 100
x = np.linspace(0, 10, N)
y = model(*params_true, *const, x) + np.random.normal(0, 2, N)
yerr = np.abs(np.random.normal(2, 0.5, (N, 2)))


# settings
params_try = np.array([5, 2, 1, 0])
params_sigma = np.array([2, 3, 0.1, 0.1])
init = np.array([params_try, params_sigma]).T

nwalkers = 400
nsteps = 200
amputate = int(0.5 * nsteps)


# mcmc 
data_params = {
        'x' : x, 
        'y' : y, 
        'yerr' : yerr
        }
model_params = {
        'model' : model, 
        'init' : init, 
        'prior_data' : prior_data, 
        'params_names' : params_names
        }
settings_params = {
        'nwalkers' : nwalkers, 
        'nsteps' : nsteps, 
        'amputate' : amputate}

mcmc_quick(data_params, model_params, settings_params, 
           prnt=True, show=True, save=False)

#mcmc(data=(x, y, yerr), model_params=(model, init, prior_data, params_names), 
#     settings=(nwalkers, nsteps, amputate), prnt=True, show=True, save=False)
