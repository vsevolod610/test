import emcee
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

# Choose the "true" parameters.
a_true = -0.9594
b_true = 4.294

# Generate some synthetic data from the model.
N = 50
x = np.sort(10 * np.random.rand(N))
yerr = 0.1 + 0.5 * np.random.rand(N)
y = a_true * x + b_true + yerr * np.random.randn(N)

def log_probability(theta, x, y, yerr):
    a, b = theta
    model = a * x + b
    sigma2 = yerr**2 
    return -0.5 * np.sum((y - model) ** 2 / sigma2)

nwalker = 32
nstep = 1000
pos = [a_true, b_true] + 1e-2 * np.random.randn(nwalker, 2)
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                args=(x, y, yerr))
sampler.run_mcmc(pos, nstep, progress=True);
#tau = sampler.get_autocorr_time()
#print(tau)
flat_samples = sampler.get_chain(discard=100, thin=1, flat=True)
print(flat_samples.shape)

###
what = sampler.chain
print(type(what), what.shape, what[-1, -1])
print(type(flat_samples), flat_samples.shape, flat_samples[-1])

#### ChainConsum
from chainconsumer import ChainConsumer
c = ChainConsumer()
c.add_chain(flat_samples, parameters=['m', 'b'])
summary = c.analysis.get_summary()
fig0 = c.plotter.plot(legend=False, figsize=(6, 6))
fig1 = c.plotter.plot_walks(convolve=100, figsize=(6, 6))
print(summary)

plt.show()
