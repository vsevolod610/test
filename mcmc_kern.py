# -*- coding: utf-8 -*-
"""
MCMC kern
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
from multiprocessing import Pool


# MCMC

def prior_func(params, prior_data=None):
    if prior_data is None:
        return 0
    prior_value = 0

    # box prior
    if 'box' in list(prior_data):
        for k in list(prior_data['box']):
            left, right = prior_data['box'][k]
            if not (left < params[k] < right):
                prior_value += -np.inf

    # gauss prior
    if 'gauss' in list(prior_data):
        for k in list(prior_data['gauss']):
            mu, sigma_p, sigma_m = prior_data['gauss'][k]
            sigma = sigma_p if (params[k] - mu) else sigma_m
            prior_value += -0.5 * (params[k] - mu)**2 / sigma**2

    return prior_value 


def log_probability(params, model, x, y, yerr=None, prior_data=None):

    # priors
    if prior_data is None:
        prior_value = 0
    else:
        prior_value = prior_func(params, prior_data)
        if not np.isfinite(prior_value):
            return -np.inf

    # model
    const = []
    if type(prior_data) == dict:
        if 'const' in prior_data:
            const = list(prior_data['const'].values())

    m = model(*params, *const, x)

    # sigma
    N = len(y)
    if np.shape(yerr) == ():
        sigma2 = np.ones(N)
    if np.shape(yerr) == (N,):
        sigma2 = yerr ** 2
    if np.shape(yerr) == (N, 2):
        sigma2 = np.zeros(N)
        for i in range(N):
            sigma2[i] = (yerr[i,1] if m[i] > y[i] else yerr[i, 0]) ** 2

    # lp
    lp_value = -0.5 * np.sum([(y[i] - m[i]) ** 2 / sigma2[i] for i in range(N)])
    lp_value += prior_value
    return lp_value


def mcmc_kern(model, nwalkers, nsteps, init, x, y, yerr=None, prior_data=None):
    ndim = len(init)
    pos = init[:, 0] + init[:, 1] * np.random.randn(nwalkers, ndim)

    # check init posintion in prior-box
    for k, param in enumerate(pos):
        while not np.isfinite(prior_func(param, prior_data)):
            param = init[:, 0] + init[:, 1] * np.random.rand(ndim)
        pos[k] = param

    # mcmc mechanism
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                    log_probability, args=(model, x, y, yerr, prior_data), pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    return sampler


# Pics

def pic_chain(sampler, params_names=None):
    samples = sampler.get_chain()
    ndim = len(samples[0,0,:])
    if params_names is None:
        params_names = np.arange(ndim)

    fig, ax = plt.subplots(nrows=ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()

    # plot(chain)
    if ndim == 1:
        ax.plot(samples[:, :, 0], "k", alpha=0.3)
        ax.set_ylabel(params_names[0], fontsize=12)
        ax.set_xlabel(r'steps', fontsize=12)
    else:
        for i, row in enumerate(ax, start=0):
            row.plot(samples[:, :, i], "k", alpha=0.3)
            row.set_ylabel(params_names[i], fontsize=12)
        row.set_xlabel(r'steps', fontsize=12)
    return fig, ax


def pic_fit(sampler, model, x, y, yerr=None, prior_data=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    samples = sampler.get_chain()
    sample_last = samples[-1, :, :]
    params_chi = max(sample_last,
            key=lambda s: log_probability(s, model, x, y, yerr, prior_data))

    const = []
    if type(prior_data) == dict:
        if 'const' in prior_data:
            const = list(prior_data['const'].values())

    # plot(set of fit)
    for w in sample_last:
        ax.plot(x, model(*w, *const, x), 'b', alpha=0.09)

    # plot(data)
    if np.shape(yerr) == ():
        ax.plot(x, y, '.k', alpha=0.5, label='data')
    else:
        ax.errorbar(x, y, yerr.T, label='data',
                capsize=3.5, mew=1.5, fmt='.k', alpha=0.5)

    # plot(best fit)
    ax.plot(x, model(*params_chi, *const, x), 'r', label='best fit')

    ax.legend(frameon=False)
    return fig, ax

