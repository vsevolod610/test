# -*- coding: utf-8 -*-
"""
MCMC: quick call mcmc
""" 

import gc
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')

import matplotlib.pyplot as plt

from .mcmc_realyze import mcmc_run, mcmc_analyze, mcmc_summary
from .mcmc_pics import pic_params, pic_chain, pic_fit


def args_data(x, y, yerr=None):
    return x, y, yerr

def args_model(model, init, prior_data=None, params_names=None):
    return model, init, prior_data, params_names

def args_settings(nwalkers, nsteps, amputate=0):
    return nwalkers, nsteps, amputate


def mcmc_quick(data_params, model_params, settings_params, **kwargs):
    # args
    data = args_data(**data_params)
    model, init, prior_data, params_names = args_model(**model_params)
    nwalkers, nsteps, amputate = args_settings(**settings_params)

    # kwargs
    prnt = kwargs.get('prnt', False)
    show = kwargs.get('show', False)
    save = kwargs.get('save', False)

    nproc = None # processes limit

    # mcmc
    sampler = mcmc_run(data, model, init, nwalkers, nsteps, prior_data, nproc)
    analyze = mcmc_analyze(sampler, amputate, params_names)
    summary = mcmc_summary(analyze, prnt=prnt)

    # pics
    if show or save:
        if not save: save = [False, False, False]
        fig = pic_params(analyze, path=save[0])
        fig = pic_chain(sampler, amputate, params_names, path=save[1])
        fig = pic_fit(sampler, model, data, prior_data, path=save[2])
    if show: plt.show()

    # garved collector
    plt.close('all')
    gc.collect()

    return summary

