# -*- coding: utf-8 -*-
"""
MCMC ...
"""

import gc
import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

from mcmc.runner import mcmc_run
from mcmc.pictures import pic_chain, pic_fit

def mcmc(data, model_params, settings, **kwargs):
    # args
    x, y, yerr, *_ = *data, None
    model, init, prior_data, params_names, *_ = *model_params, None, None
    nwalkers, nsteps, amputate = settings

    sampler = mcmc_run(data, model, nwalkers, nsteps, init, 
                       prior_data=prior_data)
    summary = mcmc_analyze(sampler, data, model_params, amputate, **kwargs)

    return summary

def mcmc_analyze(sampler, data, model_params, amputate, **kwargs):
    # args
    x, y, yerr, *_ = *data, None
    model, init, prior_data, params_names, *_ = *model_params, None, None

    show = False
    save = False
    if 'prnt' in kwargs: prnt = kwargs['prnt']
    if 'show' in kwargs: show = kwargs['show']
    if 'save' in kwargs: save = kwargs['save']

    ndim = len(init) 

    samples = sampler.get_chain()
    flat_sample = sampler.chain[:, amputate : , :].reshape((-1, ndim))
    #flat_sample = sampler.get_chain(discard=amputate, thin=15, flat=True)
    #print(type(samples), samples.shape())
    c = ChainConsumer()
    c.add_chain(flat_sample, parameters=params_names)

    summary = c.analysis.get_summary()

    # print
    if prnt == True:
        s = [" {:>4}: {}".format(k, summary[k]) for k in summary.keys()]
        print("\nMCMC results:", *s, sep='\n')

    # pics
    #if 'show' in kwargs or 'save' in kwargs:
    if 'save' in kwargs or 'show' in kwargs:
        fig0 = c.plotter.plot(legend=False, figsize=(6, 6))
        fig1 = pic_chain(samples, amputate=amputate, params_names=params_names)
        fig2 = pic_fit(samples, model, x, y, yerr, prior_data)

        #fig1 = c.plotter.plot_walks(convolve=100, figsize=(6, 6))
        if save:
            fig0.savefig(save[0])
            fig1.savefig(save[1])
            fig2.savefig(save[2])

            # garved collector
            plt.close()
            gc.collect()
        if show:
            plt.show()


    return summary

