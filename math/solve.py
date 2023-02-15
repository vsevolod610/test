# -*- coding: utf-8 -*-
"""
Math: solve transcendental equation
"""

import numpy as np
from scipy.optimize import fsolve

a = 3 / np.pi
f = lambda x: np.sin(x)/x - a

print(fsolve(f, 2))
