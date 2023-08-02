# -*- coding: utf-8 -*-
"""
MCMC: pictures definition
""" 

import gc
import numpy as np

import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

from .mcmc_chi2 import log_probability


def pic_params(c, path=False):
    fig = c.plotter.plot(legend=False, figsize=(6, 6))

    if path:
        fig.savefig(path, bbox_inches='tight')

    return fig


def pic_chain(sampler, amputate=None, params_names=None, path=False):
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

    if path:
        fig.savefig(path, bbox_inches='tight')

    return fig


def pic_fit(sampler, model, data, prior_data=None, path=False):
    chain = sampler.get_chain()
    # FIXME: legacy code
    x, y, yerr, *_ = *data, None 

    # sort by x:
    idx = x.argsort()
    x = x[idx]
    y = y[idx]
    if yerr is not None: yerr = yerr[idx]


    last_step = chain[-1, :, :]
    chi2_func = lambda param: -log_probability(param, model, data, prior_data)
    params_chi = min(last_step, key=chi2_func)
    chi2_min =  chi2_func(params_chi)


    fig, ax = plt.subplots(figsize=(8, 8))

    const = []
    if type(prior_data) == dict:
        if 'const' in prior_data:
            # FIXME: order of values not gгuarantee in python < 3.7 
            const = list(prior_data['const'].values())

    # fill_between
    if 1:
        chi2_list = np.array([chi2_func(param) - chi2_min for param in last_step])
        # log: 1sigma: (0.1 - 0.48) ~ 100, 2sigma (0.55 - 2) ~ 50, 2sigma ...
        params_3sigma = last_step[chi2_list < 2]

        y3_min = [min([model(*w, *const, xi) for w in params_3sigma]) for xi in x]
        y3_max = [max([model(*w, *const, xi) for w in params_3sigma]) for xi in x]

        ax.fill_between(x, y3_min, y3_max, label=r'$3\sigma$',
                        alpha=0.6, color='b', linewidth=2, linestyle='-')

    # plot(set of fit)
    if 0:
        for w in last_step:
            ax.plot(x, model(*w, *const, x), 'b', alpha=0.5)

    # plot(data)
    N = len(y)
    if np.shape(yerr) == ():
        ax.plot(x, y, '.k', alpha=0.5, label='data')
    if np.shape(yerr) == (N,):
        ax.errorbar(x, y, yerr, label='data', 
                    capsize=3.5, mew=1.5, fmt='.k', alpha=0.5)
    if np.shape(yerr) == (N, 2):
        ax.errorbar(x, y, [yerr[:, 1], yerr[:, 0]], label='data', 
                    capsize=3.5, mew=1.5, fmt='.k', alpha=0.5)

    # plot(best fit)
    ax.plot(x, model(*params_chi, *const, x), 'r', label='best fit')

    ax.legend(frameon=False)

    if path:
        fig.savefig(path, bbox_inches='tight')

    return fig
