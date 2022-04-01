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
        # 建立 P 條染色體作為父代，並計算各自的適應值
        parent = self.initial_population()
        # 取得父代染色體的排名，並且分群
        parent, parent_front_set = self.fast_nondominated_sort(parent)
        # 計算父代染色體各群的擁擠度
        for front_idx, front in enumerate(parent_front_set):
            parent_front_set[front_idx] = self.calculate_crowding_distance(front)
        # 從父代建立子代 : 選擇 -> 交配 -> 突變
        children = self.create_children(parent)
        # 用來放 父代 + 子代 用的
        returned_family_front_set = None
        # 開始迭代
        for g in range(self.G):
            # 父代與子代合併
            family = parent + children
            # 取得父代 + 子代染色體的排名，並且分群
            family, family_front_set = self.fast_nondominated_sort(family)
            # 建立空的容器
            parent_new = []
            # 菁英策略，逐批取的群，同時計算擁擠度，直到把容器塞滿或者快滿
            front_idx = 0
            while len(parent_new) + len(family_front_set[front_idx]) <= self.P:
                self.calculate_crowding_distance(family_front_set[front_idx])
                parent_new = parent_new + family_front_set[front_idx]
                front_idx += 1
            # 計算 父代 + 子代 front_num + 1 群的擁擠度
            family_front_set[front_idx] = self.calculate_crowding_distance(family_front_set[front_idx])
            # 對 父代 + 子代 front_num + 1 群的染色體依擁擠度作排序
            family_front_set[front_idx].sort(key=lambda chromosome: chromosome.crowding_distance, reverse=True)
            # 若容器還沒滿，則用父代 + 子代 front_num + 1 群的染色體充數
            parent_new = parent_new + family_front_set[front_idx][0:self.P-len(parent_new)]
            # 把父代 + 子代備份起來
            returned_family_front_set = family_front_set
            # 容器作為新父代
            parent = parent_new
            # 取得新父代染色體的排名，並且分群
            parent, parent_front_set = self.fast_nondominated_sort(parent)
            # 計算新父代染色體各群的擁擠度
            for front_idx, front in enumerate(parent_front_set):
                parent_front_set[front_idx] = self.calculate_crowding_distance(front)
            # 建立子代 : 選擇 -> 交配 -> 突變
            children = self.create_children(parent)
        return returned_family_front_set[0]

# %% 產生初始解
    def initial_population(self):
        parent = [self.create_chromosome() for i in range(self.P)]
        parent[0].feature = np.array([200, 90000])
        parent[1].feature = np.array([190, 100000])
        parent[2].feature = np.array([180, 65000])
        parent[3].feature = np.array([170, 75000])
        parent[4].feature = np.array([160, 80000])
        parent[5].feature = np.array([150, 40000])
        parent[6].feature = np.array([145, 44000])
        parent[7].feature = np.array([140, 47000])
        parent[8].feature = np.array([135, 49000])
        parent[9].feature = np.array([130, 50000])
        parent[0].fitness = np.array([200, 90000])
        parent[1].fitness = np.array([190, 100000])
        parent[2].fitness = np.array([180, 65000])
        parent[3].fitness = np.array([170, 75000])
        parent[4].fitness = np.array([160, 80000])
        parent[5].fitness = np.array([150, 40000])
        parent[6].fitness = np.array([145, 44000])
        parent[7].fitness = np.array([140, 47000])
        parent[8].fitness = np.array([135, 49000])
        parent[9].fitness = np.array([130, 50000])
        return parent

    def create_chromosome(self):
        chromosome = Chromosome()
        chromosome.feature = np.random.uniform(low=self.lb, high=self.ub)
        chromosome.fitness = self.calculate_fitness(chromosome)
        return chromosome

# %% 快速非支配排序
    def fast_nondominated_sort(self, population):
        front_set = [[]]
        for master in population:
            master.dominated_counter = 0
            master.dominat_solutions = []
            for slave in population:
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

        return population, front_set

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
        return front

# %% 產生子代
    def create_children(self, parent):
        children = []
        while len(children) < self.P:
            p1 = self.tournament_selection(parent)
            p2 = p1
            while p1 == p2:
                p2 = self.tournament_selection(parent)
            c1, c2 = self.crossover(p1, p2)
            self.mutation(c1)
            self.mutation(c2)
            c1.fitness = self.calculate_fitness(c1)
            c2.fitness = self.calculate_fitness(c2)
            children.append(c1)
            children.append(c2)
        return children

# %% 選擇
    def tournament_selection(self, parent):
        participants = np.random.choice(parent,
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
