# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:53:30 2022

@author: ZongSing_NB2
"""

import numpy as np


class sch():
    def __init__(self):
        self.ub = [55]
        self.lb = [-55]
        self.D = 1
        self.M = 2
        self.min_problem = True

    def fitness(self, chromosome):
        f1 = chromosome.feature[0] ** 2
        f2 = (chromosome.feature[0] - 2) ** 2
        fitness = np.array([f1, f2])

        return fitness

class test():
    def __init__(self):
        self.ub = [200, 100000]
        self.lb = [100, 25000]
        self.D = 2
        self.M = 2
        self.min_problem = False

    def fitness(self, chromosome):
        f1 = chromosome.feature[0]
        f2 = chromosome.feature[1]
        fitness = np.array([f1, f2])

        return fitness
