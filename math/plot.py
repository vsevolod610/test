# -*- coding: utf-8 -*-
"""
Pic: 
"""

import gc
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: np.sin(x)
x_min, x_max = -10, 10
x = np.linspace(x_min, x_max, 1000)


fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(x, f(x))

#axes
ax.plot(x, 0*x, 'k')
ax.plot(0*x, x, 'k')
ax.axvline(x=2, color='r')

ax.set_xlim(-5, 5)
ax.set_ylim(-2, 2)

#ax.set_xlabel(r"x")
#ax.set_ylabel(r"y")
#ax.legend(frameon=False)
#ax.set_yscale('log')

#fig.savefig('pic.pdf', format='pdf', bbox_inches='tight')

plt.show()

# garved collector
plt.close()
gc.collect()
