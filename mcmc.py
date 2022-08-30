# -*- coding: utf-8 -*-
"""
MCMC
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
from multiprocessing import Pool
from chainconsumer import ChainConsumer


def log_prior_box(v, vleft, vright):
    if vleft < v < vright:
        return 0.0
    return -np.inf


def log_prior_gauss(v, mean, sigma_p, sigma_m):
    sigma = sigma_p if (v - mean) else sigma_m
    return - 0.5 * (v - mean) ** 2 / sigma ** 2


def prior(params, prior_data):
    if not prior_data:
        return 0
    prior_value = 0

    for p_k in list(prior_data['box']):
        left, right = prior_data['box'][p_k]
        prior_value += log_prior_box(params[p_k], left, right)

    for p_k in list(prior_data['gauss']):
        mu, sigmap, sigmam = prior_data['gauss'][p_k]
        prior_value += log_prior_gauss(params[p_k], mu, sigmap, sigmam)

    return prior_value 


def log_probability(params, model, x, y, yerr=0, prior_data=0):
    if not prior_data:
        prior_value = 0
    else:
        prior_value = prior(params, prior_data)
        if not np.isfinite(prior_value):
            return -np.inf

    const = []
    if type(prior_data) == dict:
        if 'const' in prior_data:
            const = list(prior_data['const'].values())

    m = model(*params, *const, x)

    N = len(y)
    if np.shape(yerr) == ():
        sigma2 = np.ones(N)
    if np.shape(yerr) == (N,):
        sigma2 = yerr ** 2
    if np.shape(yerr) == (N, 2):
        sigma2 = np.zeros(N)
        for i in range(N):
            sigma2[i] = (yerr[i,1] if m[i] > y[i] else yerr[i, 0]) ** 2

    lp_value = -0.5 * np.sum([(y[i] - m[i]) ** 2 / sigma2[i] for i in range(N)])
    lp_value += prior_value
    return lp_value


def mcmc_kern(model, nwalkers, nsteps, ndim, init, x, y, yerr=0, prior_data=0):
    pos = init[:, 0] + init[:, 1] * np.random.randn(nwalkers, ndim)

    # check init posintion in prior-box
    for k, param in enumerate(pos):
        while not np.isfinite(prior(param, prior_data)):
            param = params_try + params_sigma * np.random.rand(ndim)
        pos[k] = param

    # mcmc mechanism
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                    log_probability, args=(model, x, y, yerr, prior_data), pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    return sampler


# Pic's

def pic_chain(sampler, params_names):
    samples = sampler.get_chain()
    ndim = len(samples[0,0,:])

    fig, ax = plt.subplots(nrows=ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()

    for i, row in enumerate(ax, start=0):
        row.plot(samples[:, :, i], "k", alpha=0.3)
        row.set_ylabel(params_names[i], fontsize=12)

    row.set_xlabel(r'steps', fontsize=12)
    return fig


def pic_fit(sampler, model, x, y, yerr, prior_data):
    fig, ax = plt.subplots(figsize=(8, 8))
    samples = sampler.get_chain()
    sample_last = samples[-1, :, :]
    params_chi = max(sample_last,
            key=lambda s: log_probability(s, model, x, y, yerr, prior_data))

    const = []
    if type(prior_data) == dict:
        if 'const' in prior_data:
            const = list(prior_data['const'].values())

    for w in sample_last:
        ax.plot(x, model(*w, *const, x), 'b', alpha=0.09)

    if np.shape(yerr) == ():
        ax.plot(x, y, '.k', alpha=0.5, label='data')
    else:
        ax.errorbar(x, y, yerr.T, label='data',
                capsize=3.5, mew=1.5, fmt='.k', alpha=0.5)

    ax.plot(x, model(*params_chi, *const, x), 'r', label='best fit')

    ax.legend(frameon=False)
    return fig


def example_model(a, b, c, x0, z, x):
    return a * np.sin(c * (x - x0)) + b - z


if __name__ == "__main__":
    # model
    params_names = ['a', 'b', 'c', 'x0']
    const_names = ['z']
    model = example_model

    prior_data = dict()
    prior_data['box'] = {params_names.index('a'): [2, 8],
                         params_names.index('b'): [1, 4]}
    prior_data['gauss'] = {params_names.index('b'): [2, 0.1, 0.1]}
    prior_data['const'] = {const_names.index('z'): 0.0}
    const = list(prior_data['const'].values())

    # data
    params_true = [5, 2, 1, 0]
    N = 100
    x = np.linspace(0, 10, N)
    y = model(*params_true, *const, x) + np.random.normal(0, 2, N)
    yerr = np.abs(np.random.normal(2, 0.5, (N, 2)))

    # settings
    ndim = len(params_names)

    params_try = np.array([5, 2, 1, 0])
    params_sigma = np.array([2, 3, 0.1, 0.1])
    init = np.array([params_try, params_sigma]).T

    nwalkers = 100
    nsteps = 200
    amputete = int(0.5 * nsteps)

    # mcmc
    sampler = mcmc_kern(model, nwalkers, nsteps, ndim, init, x, y, yerr, prior_data)

    flat_sample = sampler.chain[:, amputete : , :].reshape((-1, ndim))
    c = ChainConsumer()
    c.add_chain(flat_sample, parameters=params_names)

    summary = c.analysis.get_summary(parameters=params_names)
    print("\nMCMC results:")
    print(*[" {:>4}: {}".format(k, summary[k]) for k in summary.keys()], sep='\n')

    # Pics
    fig = pic_chain(sampler, params_names)
    fig = c.plotter.plot(display=False, legend=False, figsize=(6, 6))
    fig = pic_fit(sampler, model, x, y, yerr, prior_data)

    plt.show()
