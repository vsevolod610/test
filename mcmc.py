# -*- coding: utf-8 -*-
""""
set of T0_i temperature --> Estimate T0

require:    ../Result/Result.txt
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"



import numpy as np
#import matplotlib.pyplot as plt
import emcee
from chainconsumer import ChainConsumer


params_names = ['a', 'b']
ndim = len(params_names)

def model(a, b, x):
    return a * x + b

# data
a_true = 5
b_true = 2
N = 50
x = np.linspace(0, 10, N)
y = model(a_true, b_true, x) + np.random.normal(0, 2, N)
yerr = np.abs(np.random.normal(2, 0.5, (N, 2)))



#____________________________________mcmc_________________________________


def log_probability(params, x, y, yerr):
    a, b = params
    m = model(a, b, x)
    sigma2 =  np.array([(yerr[i, 1] if m[i] > yi else yerr[i, 0]) for i, yi in enumerate(y)]) ** 2
    return - 0.5 * np.sum([(yi - m[i]) ** 2 / sigma2[i] for i, yi in enumerate(y)])

# setting of mcmc
nwalkers = 100
nsteps = 200
amputete = int(0.5 * nsteps)
pos = [a_true, b_true] + np.random.randn(nwalkers, ndim)
# mcmc mechanism
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
sampler.run_mcmc(pos, nsteps, progress=True)
flat_sample = sampler.chain[:, amputete : , :].reshape((-1, ndim))
c = ChainConsumer()
c.add_chain(flat_sample, parameters=params_names)
summary = c.analysis.get_summary(parameters=params_names)
print(summary)

'''
#____________________________________pic__________________________________
# MCchain pic
fig, ax = plt.subplots(figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
ax.plot(samples[:, :, 0], "k", alpha=0.3)
ax.set_xlabel(r'steps', fontsize=12)
ax.set_ylabel(r'$T_0$', fontsize=12)


# chainconsum pic
fig = c.plotter.plot(display=False, legend=False, figsize=(6, 6))

#pic
fig, ax = plt.subplots(figsize=(8, 8))
ax.errorbar(x, y, np.transpose(yerr), capsize=3.5, mew=1.5, fmt='.k', alpha=0.5)
ax.plot(x, model(a_true, b_true, x), 'k')
a_fit = summary['a']
b_fit = summary['b']
ax.plot(x, model(a_fit[1], b_fit[1], x))
ax.fill_between(x, model(a_fit[0], b_fit[0], x), model(a_fit[2], b_fit[2], x),
            facecolor='b', alpha=0.3, color='b', linewidth=2, linestyle='-')

'''