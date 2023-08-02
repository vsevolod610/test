# -*- coding: utf-8 -*-
"""
MCMC: quick
""" 

import gc
import numpy as np

#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from .mcmc_run import mcmc_run
from .mcmc_analyze import mcmc_analyze, mcmc_summary
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
    prnt = kwargs['prnt'] if 'prnt' in kwargs else False
    show = kwargs['show'] if 'show' in kwargs else False
    save = kwargs['save'] if 'save' in kwargs else False
    if not save: save = [False, False, False] # if call save=Flase

    # mcmc
    sampler = mcmc_run(data, model, init, nwalkers, nsteps, prior_data)
    analyze = mcmc_analyze(sampler, amputate, params_names)
    summary = mcmc_summary(analyze, prnt=prnt)

    # pics
    fig = pic_params(analyze, save_path=save[0])
    fig = pic_chain(sampler, amputate, params_names, save_path=save[1])
    fig = pic_fit(sampler, model, data, prior_data, save_path=save[2])
    if show: plt.show()

    # garved collector
    plt.close('all')
    gc.collect()

    return summary

