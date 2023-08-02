# -*- coding: utf-8 -*-
"""
MCMC: chi2 definition
""" 

import numpy as np


def prior_func(params, prior_data=None):
    prior_value = 0
    if prior_data is None: return 0

    # box prior
    if 'box' in prior_data:
        for k in prior_data['box']:
            left, right = prior_data['box'][k]
            if not (left < params[k] < right):
                return -np.inf

    # gauss prior
    if 'gauss' in prior_data:
        for k in prior_data['gauss']:
            mu, sigma_p, sigma_m = prior_data['gauss'][k]
            sigma = sigma_p if (params[k] - mu) else sigma_m
            prior_value += -0.5 * (params[k] - mu)**2 / sigma**2

    return prior_value 


def log_probability(params, model, data, prior_data=None):
    # FIXME: legacy code
    # args managemts: data = (x, y, yerr=None)
    x, y, yerr, *_ = *data, None

    # priors
    if prior_data is None:
        prior_value = 0
    else:
        prior_value = prior_func(params, prior_data)
        if not np.isfinite(prior_value):
            return -np.inf

    # model
    const = []
    if type(prior_data) == dict:
        if 'const' in prior_data:
            # FIXME: order of values not gгuarantee in python < 3.7 
            const = list(prior_data['const'].values())

    m = model(*params, *const, x)

    # sigma
    N = len(y)
    if np.shape(yerr) == ():
        sigma2 = np.ones(N)
    if np.shape(yerr) == (N,):
        sigma2 = yerr ** 2
    if np.shape(yerr) == (N, 2):
        sigma2 = (m <= y) * yerr[:, 1]**2 + (m > y) * yerr[:, 0]**2

    # lp
    lp_value = -0.5 * np.sum((y - m) ** 2 / sigma2)
    #lp_value += -0.5 * np.sum(np.log(sigma2))
    lp_value += prior_value
    return lp_value

