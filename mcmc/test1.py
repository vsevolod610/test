# -*- coding: utf-8 -*-
"""
MCMC analyze example "Sin"
"""

import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

from mcmc_kern import mcmc_kern
from mcmc_analyze import pic_chain, pic_fit
from mcmc_analyze import mcmc_analyze


np.random.seed(123)


def example_model(a, b, c, x0, z, x):
    return a * np.sin(c * (x - x0) ** z) + b 


if __name__ == "__main__":
    # model
    params_names = ['a', 'b', 'c', 'x0']
    const_names = ['z']
    model = example_model
    ndim = len(params_names)

    prior_data = dict()
    prior_data['box'] = {params_names.index('a'): [2, 8],
                         params_names.index('b'): [1, 4]}
    prior_data['gauss'] = {params_names.index('b'): [2, 0.1, 0.1]}
    prior_data['const'] = {const_names.index('z'): 1.0}

    const = list(prior_data['const'].values())

    # data
    params_true = [5, 2, 1, 0]
    N = 100
    x = np.linspace(0, 10, N)
    y = model(*params_true, *const, x) + np.random.normal(0, 2, N)
    yerr = np.abs(np.random.normal(2, 0.5, (N, 2)))

    # settings
    params_try = np.array([5, 2, 1, 0])
    params_sigma = np.array([2, 3, 0.1, 0.1])
    init = np.array([params_try, params_sigma]).T

    nwalkers = 100
    nsteps = 200
    amputate = int(0.5 * nsteps)

    # mcmc 
    sampler = mcmc_kern(model, nwalkers, nsteps, init, x, y, yerr, prior_data)
    mcmc_analyze(sampler, amputate, params_names, prnt=True, pic=True)
    fig, ax = pic_chain(sampler, params_names)
    fig, ax = pic_fit(sampler, model, x, y, yerr, prior_data)

    plt.show()

