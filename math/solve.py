# -*- coding: utf-8 -*-
"""
Math: solve transcendental equation
"""

import numpy as np
from scipy.optimize import fsolve

a = 3 / np.pi
f = lambda x: np.sin(x)/x - a
print(*fsolve(f, 2))


"""
Math: find minimum
"""

import numpy as np
from scipy.optimize import minimize

f = lambda x: -1 * x * np.exp(-x)
x_start = 1
soln = minimize(f, x_start)
print(*soln.x)


"""
Math: numerical integrate
"""

import numpy as np
from scipy import integrate

f = lambda x: np.exp(-x ** 2)
x_start = -np.Inf
x_stop = np.Inf
solv = integrate.quad(f, x_start, x_stop)
print("{} +- {}".format(*solv))
