# -*- coding: utf-8 -*-
"""
MCMC all in one file
""" 

import gc
import emcee
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from chainconsumer import ChainConsumer


def prior_func(params, prior_data=None):
    prior_value = 0
    if prior_data is None: return 0

    # box prior
    if 'box' in prior_data:
        for k in prior_data['box']:
            left, right = prior_data['box'][k]
            if not (left < params[k] < right):
                return -np.inf

    # gauss prior
    if 'gauss' in prior_data:
        for k in prior_data['gauss']:
            mu, sigma_p, sigma_m = prior_data['gauss'][k]
            sigma = sigma_p if (params[k] - mu) else sigma_m
            prior_value += -0.5 * (params[k] - mu)**2 / sigma**2

    return prior_value 


def log_probability(params, model, data, prior_data=None):
    # args managemts: data = (x, y, yerr=None)
    x, y, yerr, *_ = *data, None

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


def mcmc_run(data, model, settings, init, prior_data=None):
    # args managment
    x, y, yerr, *_ = *data, None 
    nwalkers, nsteps = settings
    ndim = len(init)

    # check init posintion in prior-box
    pos = init[:, 0] + init[:, 1] * np.random.randn(nwalkers, ndim)
    for k, param in enumerate(pos):
        while not np.isfinite(prior_func(param, prior_data)):
            param = init[:, 0] + init[:, 1] * np.random.rand(ndim)
        pos[k] = param

    # mcmc mechanism
    # processes
    with Pool(processes=4) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(model, data, prior_data), 
                                        pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    chain = sampler.get_chain()
    
    return chain


def pic_chain(chain, amputate=None, params_names=None):
    _, _, ndim = np.shape(chain)
    if params_names is None:
        params_names = np.arange(ndim)

    fig, ax = plt.subplots(nrows=ndim, figsize=(10, 7), sharex=True)

    # plot(chain)
    if ndim == 1:
        ax.plot(chain[:, :, 0], "k", alpha=0.3)
        if amputate: ax.axvline(x=amputate, color='r')
        ax.set_ylabel(params_names[0], fontsize=12)
        ax.set_xlabel(r'steps', fontsize=12)
    else:
        for i, row in enumerate(ax, start=0):
            row.plot(chain[:, :, i], "k", alpha=0.3)
            if amputate: row.axvline(x=amputate, color='r')
            row.set_ylabel(params_names[i], fontsize=12)
        row.set_xlabel(r'steps', fontsize=12)
    return fig


def pic_fit(chain, model, data, prior_data=None):
    # args managment
    x, y, yerr, *_ = *data, None 

    fig, ax = plt.subplots(figsize=(8, 8))
    last_step = chain[-1, :, :]
    func = lambda param: log_probability(param, model, data, prior_data)
    params_chi = max(last_step, key=func)

    const = []
    if type(prior_data) == dict:
        if 'const' in prior_data:
            const = list(prior_data['const'].values())

    # plot(set of fit)
    for w in last_step:
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
    return fig


def mcmc_analyze(chain, data, model, init, prior_data, amputate, params_names, 
                 **kwargs):
    # args management
    x, y, yerr, *_ = *data, None
    ndim = len(init) 

    # kwargs
    prnt = kwargs['prnt'] if 'prnt' in kwargs else False
    show = kwargs['show'] if 'show' in kwargs else False
    save = kwargs['save'] if 'save' in kwargs else False

    flat_chain = chain[amputate :, : , :].reshape((-1, ndim))
    c = ChainConsumer()
    c.add_chain(flat_chain, parameters=params_names)
    summary = c.analysis.get_summary()

    # print
    if prnt:
        s = [" {:>4}: {}".format(k, summary[k]) for k in summary.keys()]
        print("\nMCMC results:", *s, sep='\n')

    # pics
    if show or save:
        fig0 = c.plotter.plot(legend=False, figsize=(6, 6))
        fig1 = pic_chain(chain, amputate=amputate, params_names=params_names)
        fig2 = pic_fit(chain, model, data, prior_data)

        #fig1 = c.plotter.plot_walks(convolve=100, figsize=(6, 6))
        if save:
            fig0.savefig(save[0], bbox_inches='tight')
            fig1.savefig(save[1], bbox_inches='tight')
            fig2.savefig(save[2], bbox_inches='tight')

            # garved collector
            plt.close('all')
            gc.collect()
        if show:
            plt.show()

    return summary


def mcmc(data, model_params, settings, **kwargs):
    # args management
    x, y, yerr, *_ = *data, None
    model, init, prior_data, params_names, *_ = *model_params, None, None
    nwalkers, nsteps, amputate = settings

    chain = mcmc_run(data, model, (nwalkers, nsteps), init, prior_data)
    summary = mcmc_analyze(chain, data, model, init, prior_data, amputate, 
                           params_names, **kwargs)

    return summary

