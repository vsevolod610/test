# -*- coding: utf-8 -*-
"""
MCMC: generate chain
""" 

import emcee
import numpy as np

from multiprocessing import Pool

from mcmc_chi2 import log_probability, prior_func


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
    #with Pool() as pool:
    with Pool(processes=4) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(model, data, prior_data), 
                                        pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    #chain = sampler.get_chain()
    
    return sampler

