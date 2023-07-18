# -*- coding: utf-8 -*-
"""
MCMC: quick
""" 

import gc
import emcee
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from multiprocessing import Pool
from chainconsumer import ChainConsumer

from mcmc_run import mcmc_run
from mcmc_analyze import mcmc_analyze


def mcmc(data, model_params, settings, **kwargs):
    # args management
    x, y, yerr, *_ = *data, None
    model, init, prior_data, params_names, *_ = *model_params, None, None
    nwalkers, nsteps, amputate = settings

    sampler = mcmc_run(data, model, (nwalkers, nsteps), init, prior_data)
    summary = mcmc_analyze(sampler, data, model, init, prior_data, amputate, 
                           params_names, **kwargs)

    return summary

