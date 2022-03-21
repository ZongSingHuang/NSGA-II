# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:53:30 2022

@author: ZongSing_NB2
"""

import numpy as np


class sch:
    def __init__(self):
        self.XMAX = 55
        self.XMIN = -55
        self.D = 1
        self.min_problem = True

    def fitness(self, X):
        f1 = X**2
        f2 = (X - 2)**2
        F = np.vstack([f1, f2]).T
        return F
