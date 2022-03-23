# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:01:54 2022

@author: ZongSing_NB2
"""

import numpy as np
import pandas as pd

import time


class NSGAII:
    def __init__(self,
                 benchmark,
                 P=100,
                 G=1000,
                 tour_k=2,
                 tour_prob=0.9,
                 cross_param=2,
                 mutation_param=5,
                 min_problem=True):
        self.P = P
        self.G = G
        self.tour_k = tour_k
        self.tour_prob = tour_prob
        self.benchmark = benchmark
        self.D = benchmark.D
        self.fitness = benchmark.fitness
        self.min_problem = benchmark.min_problem
        self.ub = benchmark.ub
        self.lb = benchmark.lb
        self.M = benchmark.M
        self.cross_param = cross_param
        self.mutation_param = mutation_param

        self.X_gbest = np.zeros([self.D])
        self.F_gbest = np.inf

    def opt(self):
        st = time.time()
        self.X = self.initial_population(self.P)
        self.X = self.fitness(self.X)
        self.X, front_set = self.fast_nondominated_sort(self.X)
        self.X = self.calculate_crowding_distance(self.X,
                                                  front_set)
        self.Xc = self.create_children(self.X,
                                       front_set)
        print(time.time() - st)
        return 0

# %%
    def initial_population(self, P=1):
        X = []
        for i in range(P):
            x = {'X': np.random.uniform(low=self.ub,
                                        high=self.lb,
                                        size=self.D)}
            X.append(x)
        # [-0.414, 0.467, 0.818, 1.735, 3.210, -1.272, -1.508, -1.832, -2.161, -4.105]
        # [0.467, 1.735, 0.818, -0.414, 3.210, -1.272, -1.508, -1.832, -2.161, -4.105]
        # for i in [-0.414, 0.467, 0.818, 1.735, 3.210, -1.272, -1.508, -1.832, -2.161, -4.105]:
        #     x = {'X': np.array([i])}
        #     X.append(x)
        # self.P = 10
        return pd.DataFrame(X)

    def tournament_selection(self, X):
        participants = np.random.choice(self.P,
                                        size=self.tour_k,
                                        replace=False)
        best = None
        for _, participant in X.loc[participants].iterrows():
            r = np.random.uniform()
            if (best is None) or (self.crowding_operator(participant, best) and (r <= self.tour_prob)):
                best = participant
        return best

    def crossover(self, p1, p2):
        c = self.initial_population(2)
        c1 = c.loc[0]
        c2 = c.loc[1]
        for gene_idx in range(self.D):
            u = np.random.uniform()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (self.cross_param + 1))
            else:
                beta = (2 * (1 - u)) ** (-1 / (self.cross_param + 1))

            gene1 = (p1['X'][gene_idx] + p2['X'][gene_idx]) / 2
            gene2 = np.abs((p1['X'][gene_idx] - p2['X'][gene_idx]) / 2)
            c1['X'][gene_idx] = gene1 + beta * gene2
            c2['X'][gene_idx] = gene1 - beta * gene2
        return c1, c2

    def mutation(self, c1):
        for gene_idx in range(self.D):
            u = np.random.uniform()
            if u < 0.5:
                delta = (2 * u) ** (1 / (self.mutation_param + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (self.mutation_param + 1))

            if u < 0.5:
                c1['X'][gene_idx] += delta * (c1['X'][gene_idx] - self.lb[gene_idx])
            else:
                c1['X'][gene_idx] += delta * (self.ub[gene_idx] - c1['X'][gene_idx])

            c1['X'][gene_idx] = np.clip(c1['X'][gene_idx], self.lb[gene_idx], self.ub[gene_idx])
        return c1

# %%
    def fast_nondominated_sort(self,
                               X):
        front_set = [[]]
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
                front_set[0].append(x1_idx)

        i = 0
        while len(front_set[i]):
            front = []
            for master_idx in front_set[i]:
                for slave_idx in X[master_idx]['dominat_solutions']:
                    X[slave_idx]['dominated_counter'] -= 1
                    if not X[slave_idx]['dominated_counter']:
                        X[slave_idx]['rank'] = i + 1
                        front.append(slave_idx)
            i = i + 1
            front_set.append(front)

        return pd.DataFrame(X), front_set

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
                                    X,
                                    front_set):
        crowding_distance = np.zeros(self.P)
        for front in front_set:
            front_size = len(front)
            if front_size:
                front_F = pd.DataFrame(X.loc[front, 'F'].tolist(), index=X.loc[front, 'F'].index)
                for m_idx in range(self.M):
                    sorted_F = front_F[m_idx].sort_values()
                    sorted_idx = sorted_F.index
                    crowding_distance[sorted_idx[0]] = np.inf
                    crowding_distance[sorted_idx[front_size-1]] = np.inf
                    m_values = front_F[m_idx]
                    scale = m_values.max() - m_values.min()
                    if not scale:
                        scale = 1
                    for i in range(1, front_size-1):
                        crowding_distance[sorted_idx[i]] += (sorted_F.loc[sorted_idx[i+1]] - sorted_F.loc[sorted_idx[i-1]]) / scale
        self.X['crowding_distance'] = crowding_distance
        return self.X

    def crowding_operator(self, participant, best):
        if (participant['rank'] < best['rank']) or ((participant['rank'] == best['rank']) and (participant['crowding_distance'] > best['crowding_distance'])):
            return True
        else:
            return False

# %%
    def create_children(self,
                        X,
                        front_set):
        children = pd.DataFrame()
        while len(children) < self.P:
            p1 = self.tournament_selection(X)
            p2 = p1
            while p1.equals(p2):
                p2 = self.tournament_selection(X)
            c1, c2 = self.crossover(p1, p2)
            c1 = self.mutation(c1)
            c2 = self.mutation(c2)
            c1 = self.fitness(c1)
            c2 = self.fitness(c2)
            children = pd.concat([children, c1])
            children = pd.concat([children, c2])
        return children
