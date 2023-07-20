# -*- coding: utf-8 -*-
"""
Pic: plot
"""

import gc
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: np.sin(x)
x_min, x_max = -10, 10
x = np.linspace(x_min, x_max, 1000)


fig, ax = plt.subplots(figsize=(8, 8))


# plot
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.plot(x, f(x), label='line')
ax.fill_between(x, f(1.1 * x) , f(0.9 * x),
                alpha=0.3, color='b', linewidth=2, linestyle='-')

# settings
ax.set_xlim(-5, 5)
ax.set_ylim(-2, 2)

ax.set_xlabel(r"x")
ax.set_ylabel(r"y")
ax.legend(frameon=False)
#ax.set_yscale('log')

#fig.savefig('pic.pdf', format='pdf', bbox_inches='tight')

plt.show()

# garved collector
plt.close()
gc.collect()
