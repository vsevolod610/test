# -*- coding: utf-8 -*-
"""
MCMC kern
"""

import numpy as np
import matplotlib.pyplot as plt

from mcmc_kern import log_probability

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

