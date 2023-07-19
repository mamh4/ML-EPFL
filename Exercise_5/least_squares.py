# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    a = np.dot(tx, tx.T)
    b = np.dot(tx, y)
    w = np.linalg.solve(a, b)
    mse = 1/2 * np.mean((y - np.dot(tx.T, w))**2) 
    return mse, w
