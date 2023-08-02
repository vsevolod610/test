# -*- coding: utf-8 -*-
"""
MCMC: generate & analyze chain
""" 

import emcee
import numpy as np

from multiprocessing import Pool
from chainconsumer import ChainConsumer

from .mcmc_chi2 import log_probability, prior_func


def mcmc_run(data, model, init, nwalkers, nsteps, prior_data=None, nproc=None):
    ndim = len(init)

    # check init posintion in prior-box
    pos = init[:, 0] + init[:, 1] * np.random.randn(nwalkers, ndim)
    for k, param in enumerate(pos):
        while not np.isfinite(prior_func(param, prior_data)):
            param = init[:, 0] + init[:, 1] * np.random.rand(ndim)
        pos[k] = param

    # mcmc mechanism
    with Pool(processes=nproc) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(model, data, prior_data), 
                                        pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    #chain = sampler.get_chain()
    
    return sampler


def mcmc_analyze(sampler, amputate=0, params_names=None):
    flat_chain = sampler.get_chain(discard=amputate, flat=True)
    c = ChainConsumer()
    c.add_chain(flat_chain, parameters=params_names)
    return c


def mcmc_summary(c, prnt=False):
    summary = c.analysis.get_summary()

    # print
    if prnt:
        s = [" {:>4}: {}".format(k, summary[k]) for k in summary.keys()]
        print("\nMCMC results:", *s, sep='\n')

    return summary

