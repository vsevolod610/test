# -*- coding: utf-8 -*-
"""
MCMC analyze example "Flat"
"""

import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

from mcmc_kern import mcmc_kern, log_probability
from mcmc_analyze import pic_chain, pic_fit


#np.random.seed(123)


def iden(p, x):
    return p + 0 * x


if __name__ == "__main__":
    # model
    ndim = 1
    model = iden

    # data
    N = 100
    mu, sigma, mu_s, sigma_s = 0, 1, 1, 0.5

    x = np.arange(N)
    y = np.random.normal(mu, sigma, N)
    yerr = np.abs(np.random.normal(mu_s, sigma_s, (N, 2)))

    # settings
    params_try = [1]
    params_sigma = [1]
    init = np.array([params_try, params_sigma]).T

    nwalkers = 100
    nsteps = 200
    amputete = int(0.3 * nsteps)

    # mcmc realyze
    sampler = mcmc_kern(model, nwalkers, nsteps, init, x, y, yerr)

    # mcmc analyze
    flat_sample = sampler.chain[:, amputete : , :].reshape((-1, ndim))
    c = ChainConsumer()
    c.add_chain(flat_sample)

    #summary 
    summary = c.analysis.get_summary()
    print("\nMCMC results:")
    print(*[" {:>4}: {}".format(k, summary[k]) for k in summary.keys()], sep='\n')

    # Pics
    fig, ax = pic_chain(sampler)
    fig = c.plotter.plot(display=False, legend=False, figsize=(6, 6))
    fig, ax = pic_fit(sampler, model, x, y, yerr)

    plt.show()

    from scipy.optimize import minimize
    nll = lambda *args: -log_probability(*args)
    initial = np.array([mu]) + 0.1 * np.random.randn(1)
    soln = minimize(nll, initial, args=(model, x, y, yerr))
    m = soln.x
    print(m)

