# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:41:23 2022

@author: Vsevolod
"""
from multiprocessing import cpu_count

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))



import os

os.environ["OMP_NUM_THREADS"] = "1"




import time
import numpy as np


def log_prob(theta):
    t = time.time() + np.random.uniform(0.005, 0.008)
    while True:
        if time.time() >= t:
            break
    return -0.5 * np.sum(theta ** 2)




import emcee

np.random.seed(42)
initial = np.random.randn(32, 5)
nwalkers, ndim = initial.shape
nsteps = 100

'''
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
start = time.time()
sampler.run_mcmc(initial, nsteps, progress=True)
end = time.time()
serial_time = end - start
print("\n Serial took {0:.1f} seconds".format(serial_time))
'''

from multiprocessing import Pool


start = time.time()
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
    sampler.run_mcmc(initial, nsteps, progress=True)

end = time.time()
multi_time = end - start

print("Multiprocessing took {0:.1f} seconds".format(multi_time))
#print("{0:.1f} times faster than serial".format(serial_time / multi_time))
    