# -*- coding: utf-8 -*-
"""
Math: matrics
"""

import numpy as np
from numpy import linalg
import pprint

A = np.array([
    [2, 1, -1],
    [1, 2,  1], 
    [0, -1, 3]
    ])

#print('\nMatrix: \n',A,'\n')

# det
det = linalg.det(A)
print('det: \n', det, '\n')

# собственные числа и собственные вектора
c, v = linalg.eig(A)

# упорядоченные

print("Eigen values and vectors:", *[(c[i], ' : ', list(v[i])) for i in range(len(c))], sep=' \n ')

# обратная матрица
iA = linalg.inv(A)
print("\nInverse Matrix: \n", iA,'\n')


"""
Math: matrix prod
"""

A = np.array([
    [2, 1, -1],
    [1, 2,  1], 
    [0, -1, 3]
    ])
B = np.array([
    [1, 0, 1],
    [1, 2, 1], 
    [-2, 1, 3]
    ])

#print('\nMatrix A: \n',A,)
#print('\nMatrix B: \n',B,'\n')
C = np.dot(A, B)
print('Matrix C = A.B:\n', C,'\n')


"""
Math: lin solve
"""

A = np.array([
    [2, 1, -1],
    [1, 2,  1], 
    [0, -1, 3]
    ])

b = np.array([1, 2, 3])


#print('\nMatrix A: \n',A,)
#print('\nvector b: \n',b,'\n')

x = linalg.solve(A, b)

print('Sove A.x = b:\n', x,'\n')
