# -*- coding: utf-8 -*-
"""
MCMC ...
"""

import gc
import numpy as np
import matplotlib.pyplot as plt

from mcmc.probability import log_probability


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
    return fig, ax


def pic_fit(chain, model, x, y, yerr=None, prior_data=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    last_step = chain[:, -1, :]
    func = lambda param: log_probability(param, model, x, y, yerr, prior_data)
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
    return fig, ax

