# -*- coding: utf-8 -*-
"""
MCMC analysys
    - добавить отладочные штуки: рисовать chi2
    - ndim = len(samples[0,0,:]) - так не годится определять ndim
    - params_names -> parameters
    - amputate = 0.5 * steps -> 0.5 (<1), 100 (>1) ?
"""

import gc
import numpy as np
import matplotlib.pyplot as plt

from chainconsumer import ChainConsumer

from mcmc.kern import log_probability


def mcmc_analyze(sampler, amputate, params_names=None):
    samples = sampler.get_chain()
    ndim = len(samples[0,0,:])

    flat_sample = sampler.chain[:, amputate : , :].reshape((-1, ndim))
    c = ChainConsumer()
    c.add_chain(flat_sample, parameters=params_names)

    return c

def mcmc_summary(c, prnt=False):
    summary = c.analysis.get_summary()

    # print
    if prnt is True:
        s = [" {:>4}: {}".format(k, summary[k]) for k in summary.keys()]
        print("\nMCMC results:", *s, sep='\n')

    return summary


paths_default = ['mcmc_walkers.png', 'mcmc_dist.png', 'mcmc_plot.png']

def mcmc_pics(sampler, c, model, data, amputate=None, prior_data=None, 
              params_names=None, mode='show', paths=paths_default):

    x, y, *yerr = data
    if yerr == []: yerr = None
    else: yerr = yerr[0]

    fig0 = c.plotter.plot(display=False, legend=False, figsize=(6, 6))
    #fig1 = c.plotter.plot_walks(display=False, convolve=100, figsize=(6, 6))
    fig1, ax1 = pic_chain(sampler, amputate=amputate, 
                          params_names=params_names)
    fig2, ax2 = pic_fit(sampler, model, x, y, yerr, prior_data)

    if mode == 'show':
        plt.show()
    if mode == 'save':
        fig0.savefig(paths[0])
        fig1.savefig(paths[1])
        fig2.savefig(paths[2])

        # garved collector
        plt.clf()
        plt.close()
        plt.close(fig0)
        plt.close(fig1)
        plt.close(fig2)
        plt.close('all')
        del fig0
        del fig1
        del fig2
        gc.collect()

# Pics

def pic_chain(sampler, amputate=None, params_names=None):
    samples = sampler.get_chain()
    ndim = len(samples[0,0,:])
    if params_names is None:
        params_names = np.arange(ndim)

    fig, ax = plt.subplots(nrows=ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()

    # plot(chain)
    if ndim == 1:
        ax.plot(samples[:, :, 0], "k", alpha=0.3)
        if amputate:
            ax.axvline(x=amputate, color='r')
        ax.set_ylabel(params_names[0], fontsize=12)
        ax.set_xlabel(r'steps', fontsize=12)
    else:
        for i, row in enumerate(ax, start=0):
            row.plot(samples[:, :, i], "k", alpha=0.3)
            if amputate:
                row.axvline(x=amputate, color='r')
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

