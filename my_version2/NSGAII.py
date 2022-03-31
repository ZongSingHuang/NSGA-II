# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:01:54 2022

@author: ZongSing_NB2
"""

from chromosome import Chromosome
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
        self.calculate_fitness = benchmark.fitness
        self.min_problem = benchmark.min_problem
        self.ub = benchmark.ub
        self.lb = benchmark.lb
        self.M = benchmark.M
        self.cross_param = cross_param
        self.mutation_param = mutation_param

        self.X_gbest = np.zeros([self.D])
        self.F_gbest = np.inf

    def opt(self):
        self.initial_population()
        front_set = self.fast_nondominated_sort()
        for front in front_set:
            self.calculate_crowding_distance(front)
        children = self.create_children()
        returned_population = None
        for g in range(self.G):
            self.X = self.X + children
            front_set = self.fast_nondominated_sort()
            new_population = []
            front_num = 0
            while len(new_population) + len(front_set[front_num]) <= self.P:
                self.calculate_crowding_distance(front_set[front_num])
                new_population = new_population + front_set[front_num]
                front_num += 1
            self.calculate_crowding_distance(front_set[front_num])
            front_set[front_num].sort(key=lambda chromosome: chromosome.crowding_distance, reverse=True)
            new_population = new_population + front_set[front_num][0:self.P-len(new_population)]
            returned_population = self.X
            self.X = new_population
            front_set = self.fast_nondominated_sort()
            for front in front_set:
                self.calculate_crowding_distance(front)
            children = self.create_children()
        return returned_population.fronts[0]

# %% 產生初始解
    def initial_population(self):
        self.X = [self.create_chromosome() for i in range(self.P)]
        self.X[0].feature = np.array([200, 90000])
        self.X[1].feature = np.array([190, 100000])
        self.X[2].feature = np.array([180, 65000])
        self.X[3].feature = np.array([170, 75000])
        self.X[4].feature = np.array([160, 80000])
        self.X[5].feature = np.array([150, 40000])
        self.X[6].feature = np.array([145, 44000])
        self.X[7].feature = np.array([140, 47000])
        self.X[8].feature = np.array([135, 49000])
        self.X[9].feature = np.array([130, 50000])
        self.X[0].fitness = np.array([200, 90000])
        self.X[1].fitness = np.array([190, 100000])
        self.X[2].fitness = np.array([180, 65000])
        self.X[3].fitness = np.array([170, 75000])
        self.X[4].fitness = np.array([160, 80000])
        self.X[5].fitness = np.array([150, 40000])
        self.X[6].fitness = np.array([145, 44000])
        self.X[7].fitness = np.array([140, 47000])
        self.X[8].fitness = np.array([135, 49000])
        self.X[9].fitness = np.array([130, 50000])

    def create_chromosome(self):
        chromosome = Chromosome()
        chromosome.feature = np.random.uniform(low=self.lb, high=self.ub)
        chromosome.fitness = self.calculate_fitness(chromosome)
        return chromosome

# %% 快速非支配排序
    def fast_nondominated_sort(self):
        front_set = [[]]
        for master in self.X:
            master.dominated_counter = 0
            master.dominat_solutions = []
            for slave in self.X:
                if self.dominates(master, slave):
                    master.dominat_solutions.append(slave)
                elif self.dominates(slave, master):
                    master.dominated_counter += 1
                else:
                    pass

            if not master.dominated_counter:
                master.rank = 0
                front_set[0].append(master)

        i = 0
        while len(front_set[i]):
            front = []
            for master in front_set[i]:
                for slave in master.dominat_solutions:
                    slave.dominated_counter -= 1
                    if not slave.dominated_counter:
                        slave.rank = i + 1
                        front.append(slave)
            i = i + 1
            front_set.append(front)

        return front_set

    def dominates(self,
                  p1,
                  p2):
        if self.min_problem:
            and_condition = all(p1.fitness <= p2.fitness)
            or_condition = any(p1.fitness < p2.fitness)
        else:
            and_condition = all(p1.fitness >= p2.fitness)
            or_condition = any(p1.fitness > p2.fitness)
        return and_condition and or_condition

# %% 計算擁擠度
    def calculate_crowding_distance(self,
                                    front):
        front_len = len(front)
        if front_len:
            for chromosome in front:
                chromosome.crowding_distance = 0

            for m_idx in range(self.M):
                front.sort(key=lambda chromosome: chromosome.fitness[m_idx])
                front[0].crowding_distance = np.inf
                front[-1].crowding_distance = np.inf
                m_fitness = [chromosome.fitness[m_idx] for chromosome in front]
                scale = max(max(m_fitness) - min(m_fitness), 1)
                for i in range(1, front_len-1):
                    front[i].crowding_distance += (front[i+1].fitness[m_idx] - front[i-1].fitness[m_idx]) / scale

# %% 產生子代
    def create_children(self):
        children = []
        while len(children) < self.P:
            p1 = self.tournament_selection()
            p2 = p1
            while p1 == p2:
                p2 = self.tournament_selection()
            c1, c2 = self.crossover(p1, p2)
            self.mutation(c1)
            self.mutation(c2)
            c1.fitness = self.calculate_fitness(c1)
            c2.fitness = self.calculate_fitness(c2)
            children.append(c1)
            children.append(c2)
        return children

# %% 選擇
    def tournament_selection(self):
        participants = np.random.choice(self.X,
                                        size=self.tour_k,
                                        replace=False)
        best = None
        for participant in participants:
            r = np.random.uniform()
            if (best is None) or (self.crowding_operator(participant, best) and r <= self.tour_prob):
                best = participant
        return best

    def crowding_operator(self, participant, best):
        if (participant.rank < best.rank) or (participant.rank == best.rank and participant.crowding_distance > best.crowding_distance):
            return True
        else:
            return False

# %% 交配
    def crossover(self, p1, p2):
        c1 = self.create_chromosome()
        c2 = self.create_chromosome()
        for gene_idx in range(self.D):
            beta = self.get_beta()
            gene1 = (p1.feature[gene_idx] + p2.feature[gene_idx]) / 2
            gene2 = np.abs((p1.feature[gene_idx] - p2.feature[gene_idx]) / 2)
            c1.feature[gene_idx] = gene1 + beta * gene2
            c2.feature[gene_idx] = gene1 - beta * gene2
        return c1, c2

    def get_beta(self):
        u = np.random.uniform()
        if u <= 0.5:
            beta = (2 * u) ** (1 / (self.cross_param + 1))
        else:
            beta = (2 * (1 - u)) ** (-1 / (self.cross_param + 1))
        return beta

# %% 突變
    def mutation(self, c1):
        for gene_idx in range(self.D):
            u, delta = self.get_delta()

            if u < 0.5:
                c1.feature[gene_idx] += delta * (c1.feature[gene_idx] - self.lb[gene_idx])
            else:
                c1.feature[gene_idx] += delta * (self.ub[gene_idx] - c1.feature[gene_idx])

            c1.feature[gene_idx] = np.clip(c1.feature[gene_idx], self.lb[gene_idx], self.ub[gene_idx])
        return c1

    def get_delta(self):
        u = np.random.uniform()
        if u < 0.5:
            delta = (2 * u) ** (1 / (self.mutation_param + 1)) - 1
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (self.mutation_param + 1))
        return u, delta
