# -*- coding: utf-8 -*-
"""Exercise 2.

Grid Search
"""

import numpy as np
from costs import compute_loss


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-200, 200, num_intervals)#-100
    w1 = np.linspace(-200, 150, num_intervals)#-150
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def grid_search(y, tx, grid_w0, grid_w1):
    losses = np.zeros((len(grid_w0), len(grid_w1)))
    for i, w0 in enumerate(grid_w0):
        for j, w1 in enumerate(grid_w1):
            losses[i,j] = compute_loss(y,tx,np.array([w0,w1]),method='mse')
    return losses