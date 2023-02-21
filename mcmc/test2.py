# -*- coding: utf-8 -*-
"""
MCMC analyze example "Flat"
"""

import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

from mcmc_kern import mcmc_kern, log_probability
from mcmc_analyze import pic_chain, pic_fit
from mcmc_analyze import mcmc_analyze


np.random.seed(123)


def iden(p, x):
    return p + 0 * x


if __name__ == "__main__":
    # model
    ndim = 1
    model = iden

    # data
    N = 100
    mu_true = 0
    sigma = 1
    err_mu, err_sigma = 1.0 , 0.5

    x = np.arange(N)
    y = np.random.normal(mu_true, sigma, N)
    yerr = err_mu + np.abs(np.random.normal(0.0, err_sigma, (N, 2)))
    #yerr = np.abs(np.random.normal(err_mu, err_sigma, N))

    # settings
    params_try = [0.1]
    params_sigma = [1.1]
    init = np.array([params_try, params_sigma]).T

    nwalkers = 100
    nsteps = 200
    amputate = int(0.3 * nsteps)

    # mcmc 
    sampler = mcmc_kern(model, nwalkers, nsteps, init, x, y, yerr)
    mcmc_analyze(sampler, amputate, prnt=True, pic=True)
    fig, ax = pic_chain(sampler)
    fig, ax = pic_fit(sampler, model, x, y, yerr)

    # chi2
    from scipy.optimize import minimize
    nll = lambda *args: -log_probability(*args)
    soln = minimize(nll, params_try, args=(model, x, y, yerr))
    m = soln.x
    print(m)

    fig, ax = plt.subplots(figsize=(8, 8))
    line = np.linspace(-10, 10, 100)
    nll = lambda *args: -log_probability(*args, model, x, y, yerr)
    chi2 = [nll([mu]) for mu in line]
    ax.plot(line, chi2)

    plt.show()