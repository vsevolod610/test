# -*- coding: utf-8 -*-
"""
Math: January Erevan electronic shop Problem
"""

import numpy as np
from scipy.special import comb as C

N = 300
m = 150


def prob(k, N, m):
    p_function = lambda k: C(m, k) / C(N, k) * (N - m) / (N - k)
    if type(k) is int:
        return p_funtion(k)
    if type(k) is list:
        return list(map(p_function, k))
    if type(k) is np.ndarray:
        return np.array(list(map(p_function, k)))


k = np.arange(N)
p = prob(k, 3 * N, 3 * m)

# print(k)
# print(p)

print('Sum = ', np.sum(p))
print('Mean = ', np.sum(p*k))
