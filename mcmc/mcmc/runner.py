# -*- coding: utf-8 -*-
"""
MCMC ...
"""

import numpy as np
import emcee
from multiprocessing import Pool

from mcmc.probability import log_probability, prior_func


def mcmc_run(data, model, nwalkers, nsteps, init, prior_data=None):
    x, y, yerr, *_ = *data, None 
    ndim = len(init)

    # check init posintion in prior-box
    pos = init[:, 0] + init[:, 1] * np.random.randn(nwalkers, ndim)
    for k, param in enumerate(pos):
        while not np.isfinite(prior_func(param, prior_data)):
            param = init[:, 0] + init[:, 1] * np.random.rand(ndim)
        pos[k] = param

    # mcmc mechanism
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(model, x, y, yerr, prior_data), 
                                        pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)


    return sampler

