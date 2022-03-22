# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:01:54 2022

@author: ZongSing_NB2
"""

import numpy as np
import pandas as pd


class NSGAII:
    def __init__(self,
                 benchmark,
                 P=100,
                 G=1000,
                 k=2,
                 min_problem=True):
        self.P = P
        self.G = G
        self.k = k
        self.benchmark = benchmark
        self.D = benchmark.D
        self.fitness = benchmark.fitness
        self.min_problem = benchmark.min_problem
        self.XMAX = benchmark.XMAX
        self.XMIN = benchmark.XMIN

        self.X_gbest = np.zeros([self.D])
        self.F_gbest = np.inf

    def opt(self):
        self.X = self.initial_population()
        self.X = self.fitness(self.X)
        # self.F = np.array([[160, 83300],
        #                    [200, 28800],
        #                    [200, 86100],
        #                    [160, 46000],
        #                    [180, 81500],
        #                    [170, 58600],
        #                    [190, 70800]])
        # self.P = 7
        self.F = np.array([[0.171, 5.829],
                           [0.218, 2.347],
                           [0.669, 1.396],
                           [3.011, 0.07],
                           [10.308, 10.465],
                           [1.618, 10.708],
                           [2.275, 12.308],
                           [3.355, 14.682],
                           [4.671, 17.317],
                           [16.854, 37.275]])
        self.P = 10
        front_set = self.fast_nondominated_sort(self.X)
        crowding_distance = self.calculate_crowding_distance(self.F,
                                                             front_set)
        return 0

# %%
    def initial_population(self):
        X = []
        # for i in range(self.P):
        #     x = {'X': np.random.uniform(low=self.XMIN,
        #                                 high=self.XMAX,
        #                                 size=self.D)}
        #     X.append(x)
        for i in [-0.414, 0.467, 0.818, 1.735, 3.210, -1.272, -1.508, -1.832, -2.161, -4.105]:
            x = {'X': np.array([i])}
            X.append(x)
        return pd.DataFrame(X)

# %%
    def fast_nondominated_sort(self,
                               X):
        front = [[]]
        F = X['F'].to_dict()
        X['dominat_solutions'] = float('nan')
        X['dominated_counter'] = float('nan')
        X['rank'] = float('nan')
        X = X.to_dict('records')
        for x1_idx, x1_f in F.items():
            dominat_solutions = []
            dominated_counter = 0
            for x2_idx, x2_f in F.items():
                if self.dominates(x1_f, x2_f):
                    dominat_solutions.append(x2_idx)
                elif self.dominates(x2_f, x1_f):
                    dominated_counter += 1
                else:
                    pass
            X[x1_idx]['dominat_solutions'] = dominat_solutions
            X[x1_idx]['dominated_counter'] = dominated_counter

            if not dominated_counter:
                X[x1_idx]['rank'] = 0
                front[0].append(x1_idx)

        i = 0
        while len(front[i]):
            spam = []
            for master_idx in front[i]:
                for slave_idx in X[master_idx]['dominat_solutions']:
                    X[slave_idx]['dominated_counter'] -= 1
                    if not X[slave_idx]['dominated_counter']:
                        X[slave_idx]['rank'] = i + 1
                        spam.append(slave_idx)
            i = i + 1
            front.append(spam)

        return front

    def dominates(self,
                  x1_f,
                  x2_f):
        if self.min_problem:
            and_condition = all(x1_f <= x2_f)
            or_condition = any(x1_f < x2_f)
        else:
            and_condition = all(x1_f >= x2_f)
            or_condition = any(x1_f > x2_f)
        return and_condition and or_condition

# %%
    def calculate_crowding_distance(self,
                                    F,
                                    front_set):
        crowding_distance = np.zeros(self.P)
        for front in front_set:
            front_size = len(front)
            if front_size:
                front_F = F[front]
                for m_idx in range(F.shape[1]):
                    sorted_F = front_F[front_F[:, m_idx].argsort()]
                    sorted_idx = front[front_F[:, m_idx].argsort()]
                    crowding_distance[sorted_idx[0]] = np.inf
                    crowding_distance[sorted_idx[front_size-1]] = np.inf
                    m_values = front_F[:, m_idx]
                    scale = max(m_values) - min(m_values)
                    if not scale:
                        scale = 1
                    for i in range(1, front_size-1):
                        crowding_distance[sorted_idx[i]] += (sorted_F[i+1, m_idx] - sorted_F[i-1, m_idx]) / scale
        return crowding_distance
