# -*- coding: utf-8 -*-
""""
MCMC
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
from multiprocessing import Pool
from chainconsumer import ChainConsumer


params_names = ['a', 'b', 'c']
params_try = [5, 2, 1] # need in pos
params_sigma = [2, 1, 0.1] # need in pos
ndim = len(params_names)

def model(a, b, c, x):
    return a * np.sin(c * x) + b

# data
params_true = [5, 2, 1]
N = 100
x = np.linspace(0, 10, N)
y = model(*params_true, x) + np.random.normal(0, 2, N)
yerr = np.abs(np.random.normal(2, 0.5, (N, 2)))



#____________________________________mcmc_________________________________

def log_prior_box(v, vleft, vright):
    if vleft < v < vright:
        return 0.0
    return -np.inf

def log_prior_gauss(v, mean, sigma_p, sigma_m):
    sigma = sigma_p if (v - mean) else sigma_m
    return - 0.5 * (v - mean) ** 2 / sigma ** 2

def log_probability(params, x, y, yerr, prior=lambda v: 0):
    if not np.isfinite(prior(params)):
        return -np.inf
    m = model(*params, x)
    if np.shape(yerr) == (N,):
        sigma2 = yerr ** 2
    if np.shape(yerr) == (N, 2):
        sigma2 =  np.array([(yerr[i, 1] if m[i] > yi else yerr[i, 0]) for i, yi in enumerate(y)]) ** 2
    return - 0.5 * np.sum([(yi - m[i]) ** 2 / sigma2[i] for i, yi in enumerate(y)]) + prior(params)

# setting of mcmc
nwalkers = 100
nsteps = 200
amputete = int(0.5 * nsteps)
pos = params_try + params_sigma * np.random.randn(nwalkers, ndim)

def prior(params):
    a, b, c = params
    return log_prior_box(a, 1, 8) + log_prior_gauss(b, 2, 0.1, 0.1)

# mcmc mechanism
with Pool() as pool:
    sampler = emcee.EnsembleSampler( nwalkers, ndim, log_probability, args=(x, y, yerr), pool=pool)
    sampler.run_mcmc(pos, nsteps, progress=True)
flat_sample = sampler.chain[:, amputete : , :].reshape((-1, ndim))
c = ChainConsumer()
c.add_chain(flat_sample, parameters=params_names)
summary = c.analysis.get_summary(parameters=params_names)
print(summary)

#____________________________________pic__________________________________

# MCchain pic
fig, ax = plt.subplots(nrows=ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
for i, row in enumerate(ax, start=0):
    row.plot(samples[:, :, i], "k", alpha=0.3)
    row.set_ylabel(params_names[i], fontsize=12)
row.set_xlabel(r'steps', fontsize=12)

# chainconsum pic
fig = c.plotter.plot(display=False, legend=False, figsize=(6, 6))

#pic
fig, ax = plt.subplots(figsize=(8, 8))
sample_last = samples[-1, :, :]
for w in sample_last:
    ax.plot(x, model(*w, x), 'b', alpha=0.09)
ax.errorbar(x, y, np.transpose(yerr), capsize=3.5, mew=1.5, fmt='.k', alpha=0.5)
params_fit = [summary[p][1] for p in params_names]
ax.plot(x, model(*params_fit, x), 'r')
