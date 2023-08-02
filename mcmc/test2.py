# -*- coding: utf-8 -*-
"""
MCMC analyze example "Flat"
"""

import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

from core_mcmc.mcmc_quick import mcmc_quick
from core_mcmc.mcmc_chi2 import log_probability


np.random.seed(123)


def iden(p, x):
    return p + 0 * x


# model
ndim = 1
model = iden


# data
N = 30
mu_true = 0
sigma = 1
err_mu, err_sigma = [1.0, 9.0] , 0.0

x = np.arange(N)
y = np.random.normal(mu_true, sigma, N)
yerr = err_mu + np.abs(np.random.normal(0.0, err_sigma, (N, 2)))
#yerr = np.abs(np.random.normal(err_mu, err_sigma, N))


# settings
params_try = [0.1]
params_sigma = [1.1]
init = np.array([params_try, params_sigma]).T

nwalkers = 200
nsteps = 400
amputate = int(0.3 * nsteps)


# chi2
from scipy.optimize import minimize
nll = lambda *args: -log_probability(*args)
soln = minimize(nll, params_try, args=(model, (x, y, yerr)))
m = soln.x
print('MLS: ', *m)
fig, ax = plt.subplots(figsize=(8, 8))
line = np.linspace(-10, 10, 100)
nll = lambda *args: -log_probability(*args, model, (x, y, yerr))
chi2 = [nll([mu]) for mu in line]
ax.plot(line, chi2)


# mcmc 
data_params = {
        'x' : x,
        'y' : y,
        'yerr' : yerr
        }
model_params = {
        'model' : model,
        'init' : init
        }
settings_params = {
        'nwalkers' : nwalkers,
        'nsteps' : nsteps,
        'amputate' : amputate
        }

mcmc_quick(data_params, model_params, settings_params,
          prnt=True, show=True, save=False)

