# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:53:30 2022

@author: ZongSing_NB2
"""

import numpy as np
import pandas as pd


class sch:
    def __init__(self):
        self.XMAX = 55
        self.XMIN = -55
        self.D = 1
        self.M = 2
        self.min_problem = True

    def fitness(self, X):
        X['F'] = 0
        X = X.to_dict('records')
        for idx, val in enumerate(X):
            f1 = val['X'] ** 2
            f2 = (val['X'] - 2) ** 2
            X[idx]['F'] = np.hstack([f1, f2])

        return pd.DataFrame(X)
