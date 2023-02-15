# -*- coding: utf-8 -*-
"""
Math: Int-like numbers
"""

from mpmath import mp

mp.dps = 50
n = mp.exp(mp.pi * mp.sqrt(163))

print(n)
