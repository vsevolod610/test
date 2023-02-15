# -*- coding: utf-8 -*-
"""
MCMC analyze example "Flat"
"""

import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

from mcmc_kern import mcmc_kern, pic_chain, pic_fit


def iden(p, x):
    return p + 0 * x


if __name__ == "__main__":
    # model
    ndim = 1
    model = iden

    # data
    N = 100
    x = np.arange(N)
    y = np.random.normal(0, 1, N)
    yerr = np.abs(np.random.normal(1, 0.1, (N, 2)))

    # settings
    params_try = [1]
    params_sigma = [1]
    init = np.array([params_try, params_sigma]).T

    nwalkers = 100
    nsteps = 200
    amputete = int(0.5 * nsteps)

    # mcmc realyze
    sampler = mcmc_kern(model, nwalkers, nsteps, init, x, y, yerr)

    flat_sample = sampler.chain[:, amputete : , :].reshape((-1, ndim))
    c = ChainConsumer()
    c.add_chain(flat_sample)
    summary = c.analysis.get_summary()
    print("\nMCMC results:")
    print(*[" {:>4}: {}".format(k, summary[k]) for k in summary.keys()], sep='\n')

    # Pics
    fig, ax = pic_chain(sampler)
    fig = c.plotter.plot(display=False, legend=False, figsize=(6, 6))
    fig, ax = pic_fit(sampler, model, x, y, yerr)

    plt.show()

