# -*- coding: utf-8 -*-
"""
MCMC analyze example "Dots"
"""

import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

from mcmc_kern import mcmc_kern, log_probability
from mcmc_analyze import pic_chain, pic_fit
from mcmc_analyze import mcmc_analyze


np.random.seed(123)


def line(a, b, x):
    return a * x + b


if __name__ == "__main__":
    # model
    ndim = 2
    model = line
    params_names = ['a', 'b']

    # data
    N = 100
    a_true, b_true = 1, 2
    x_min, x_max = -10, 20
    mu, sigma = 0, 1

    x = x_min + (x_max - x_min) * np.random.rand(N)
    y = model(a_true, b_true, x) + np.random.normal(mu, sigma, N)

    # settings
    params_try = [0.9, 1.2]
    params_sigma = [0.5, 1]
    init = np.array([params_try, params_sigma]).T

    nwalkers = 100
    nsteps = 200
    amputate = int(0.3 * nsteps)

    # chi2
    from scipy.optimize import minimize
    nll = lambda *args: -log_probability(*args)
    soln = minimize(nll, params_try, args=(model, x, y))
    m = soln.x
    print('MLS: ', *m)

    # mcmc 
    sampler = mcmc_kern(model, nwalkers, nsteps, init, x, y)
    mcmc_analyze(sampler, amputate, params_names, prnt=True, pic=True)
    fig, ax = pic_chain(sampler)
    fig, ax = pic_fit(sampler, model, x, y)

    plt.show()


