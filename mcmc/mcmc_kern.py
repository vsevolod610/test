# -*- coding: utf-8 -*-
"""
MCMC kern
    - init -> start_params
"""

import numpy as np
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
        sigma2 = (m <= y) * yerr[:,0] + (m > y) * yerr[:,1]

    # lp
    lp_value = -0.5 * np.sum((y - m) ** 2 / sigma2)
    #lp_value += -0.5 * np.sum(np.log(sigma2))
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
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(model, x, y, yerr, prior_data), 
                                        pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    return sampler

