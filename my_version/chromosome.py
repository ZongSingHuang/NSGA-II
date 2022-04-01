# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 08:54:57 2022

@author: zongsing.huang
"""


class Chromosome():
    def __init__(self):
        self.feature = None
        self.rank = None
        self.crowding_distance = None
        self.dominat_solutions = None
        self.dominated_counter = None
        self.fitness = None
