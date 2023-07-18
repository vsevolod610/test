# -*- coding: utf-8 -*-
"""
MCMC: pictures
""" 

import gc
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

from mcmc_chi2 import log_probability


def pic_chain(sampler, amputate=None, params_names=None):
    chain = sampler.get_chain()
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


def pic_fit(sampler, model, data, prior_data=None):
    chain = sampler.get_chain()
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
        ax.plot(x, model(*w, *const, x), 'b', alpha=0.5)

    # plot(data)
    if np.shape(yerr) == ():
        ax.plot(x, y, '.k', alpha=0.5, label='data')
    else:
        ax.errorbar(x, y, [yerr[:, 1], yerr[:, 0]], label='data', 
                    capsize=3.5, mew=1.5, fmt='.k', alpha=0.5)

    # plot(best fit)
    ax.plot(x, model(*params_chi, *const, x), 'r', label='best fit')

    ax.legend(frameon=False)
    return fig

