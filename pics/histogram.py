# -*- coding: utf-8 -*-
"""
Pic: 
"""

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))

N = 100
mu = 0
sigma = 1
data = np.random.normal(mu, sigma, N)
period = 10

ax.hist(data, period, alpha=0.7, label='')

#ax.legend(frameon=False)

plt.show()
