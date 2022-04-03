# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 08:54:57 2022

@author: zongsing.huang
"""


class Chromosome():
    def __init__(self, min_problem):
        self.feature = None
        self.rank = None
        self.crowding_distance = None
        self.dominat_solutions = None
        self.dominated_counter = None
        self.fitness = None
        self.min_problem = min_problem

    def dominates(self,
                  p2):
        and_condition = True
        or_condition = False
        if self.min_problem:
            for i, j in zip(self.fitness, p2.fitness):
                and_condition = and_condition and i <= j
                or_condition = or_condition or i < j
        else:
            for i, j in zip(self.fitness, p2.fitness):
                and_condition = and_condition and i >= j
                or_condition = or_condition or i > j
        return and_condition and or_condition
