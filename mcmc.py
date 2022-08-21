# -*- coding: utf-8 -*-
""""
MCMC
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
from multiprocessing import Pool
from chainconsumer import ChainConsumer


params_names = ['a', 'b', 'c', 'x0']
params_try = np.array([5, 2, 1, 0]) # need in pos
params_sigma = np.array([2, 3, 0.1, 0.1]) # need in pos
ndim = len(params_names)

def model(a, b, c, x0, x):
    return a * np.sin(c * (x - x0)) + b

# data
params_true = [5, 2, 1, 0]
N = 100
x = np.linspace(0, 10, N)
y = model(*params_true, x) + np.random.normal(0, 2, N)
yerr = np.abs(np.random.normal(2, 0.5, (N, 2)))

prior_data = dict()
prior_data['box'] = {'a': [2, 8],
                     'b': [1, 4]}
prior_data['gauss'] = {'b': [2, 0.1, 0.1]}


#____________________________________mcmc_________________________________

def log_prior_box(v, vleft, vright):
    if vleft < v < vright:
        return 0.0
    return -np.inf


def log_prior_gauss(v, mean, sigma_p, sigma_m):
    sigma = sigma_p if (v - mean) else sigma_m
    return - 0.5 * (v - mean) ** 2 / sigma ** 2


def prior(params, prior_data):
    a, b, c, x0 = params
    prior_value = 0

    left, right = prior_data['box']['a']
    prior_value += log_prior_box(a, left, right)

    left, right = prior_data['box']['b']
    prior_value += log_prior_box(b, left, right)

    mu, sigmap, sigmam = prior_data['gauss']['b']
    prior_value += log_prior_gauss(b, mu, sigmap, sigmam)

    return prior_value 

def log_probability(params, x, y, yerr=0, prior_data=0):
    if not prior_data:
        prior_value = 0
    else:
        prior_value = prior(params, prior_data)
        if not np.isfinite(prior_value):
            return -np.inf

    m = model(*params, x)

    N = len(y)
    if np.shape(yerr) == ():
        sigma2 == np.ones(N)
    if np.shape(yerr) == (N,):
        sigma2 = yerr ** 2
    if np.shape(yerr) == (N, 2):
        sigma2 = np.zeros(N)
        for i in range(N):
            sigma2[i] = (yerr[i,1] if m[i] > y[i] else yerr[i, 0]) ** 2

    lp_value = -0.5 * np.sum([(y[i] - m[i]) ** 2 / sigma2[i] for i in range(N)])
    lp_value += prior_value
    return lp_value


# setting of mcmc
nwalkers = 100
nsteps = 200
amputete = int(0.5 * nsteps)

pos = params_try + params_sigma * np.random.randn(nwalkers, ndim)

# check init posintion in prior-box
for k, param in enumerate(pos):
    p = param
    while not np.isfinite(prior(p, prior_data)):
        p = params_try + params_sigma * np.random.rand(ndim)
    pos[k] = p

# mcmc mechanism
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                log_probability, args=(x, y, yerr, prior_data), pool=pool)
    sampler.run_mcmc(pos, nsteps, progress=True)

flat_sample = sampler.chain[:, amputete : , :].reshape((-1, ndim))
c = ChainConsumer()
c.add_chain(flat_sample, parameters=params_names)

summary = c.analysis.get_summary(parameters=params_names)
print("\nMCMC results:")
print(*[" {:>4}: {}".format(k, summary[k]) for k in summary.keys()], sep='\n')


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

plt.show()
