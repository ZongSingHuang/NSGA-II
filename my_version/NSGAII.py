# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:01:54 2022

@author: ZongSing_NB2
"""

import random
import time

import matplotlib.pyplot as plt
import numpy as np

from chromosome import Chromosome


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

        self.X_gbest = None
        self.F_gbest = None

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

        # 用來放家族用的
        returned_family_front_set = None

        # 開始迭代
        for g in range(self.G):
            st = time.time()

            # 父代與子代合併
            family = parent + children

            # 取得家族染色體的排名，並且分群
            family, family_front_set = self.fast_nondominated_sort(family)

            # 建立空的容器
            parent_new = []

            # 菁英策略，逐批取的群，同時計算擁擠度，直到把容器塞滿或者快滿
            front_idx = 0
            while len(parent_new) + len(family_front_set[front_idx]) <= self.P:
                # 計算家族第 front_idx 群染色體的擁擠度
                self.calculate_crowding_distance(family_front_set[front_idx])
                # 把家族第 front_idx 群染色體放入新父代
                parent_new = parent_new + family_front_set[front_idx]
                front_idx += 1

            # 計算家族 front_idx + 1 群的擁擠度
            family_front_set[front_idx] = self.calculate_crowding_distance(family_front_set[front_idx])

            # 對家族 front_idx + 1 群的染色體依擁擠度作排序
            family_front_set[front_idx].sort(key=lambda chromosome: chromosome.crowding_distance, reverse=True)

            # 若容器還沒滿，則用家族 front_idx + 1 群的染色體充數
            parent_new = parent_new + family_front_set[front_idx][0:self.P-len(parent_new)]

            # 把家族備份起來
            returned_family_front_set = family_front_set

            # 父代被新父代取代
            parent = parent_new

            # 取得新父代染色體的排名，並且分群
            parent, parent_front_set = self.fast_nondominated_sort(parent)

            # 計算新父代染色體各群的擁擠度
            for front_idx, front in enumerate(parent_front_set):
                parent_front_set[front_idx] = self.calculate_crowding_distance(front)

            # 從父代建立子代 : 選擇 -> 交配 -> 突變
            children = self.create_children(parent)

            print(f'Iteration : {g}, Cost : {(time.time() - st):.2f}')

        # 從家族取得最佳解
        self.get_gbest(returned_family_front_set[0])

# %% 產生初始解
    def initial_population(self):
        parent = [self.create_chromosome() for i in range(self.P)]
        return parent

    def create_chromosome(self):
        chromosome = Chromosome(self.min_problem)
        chromosome.feature = [random.uniform(i, j) for i, j in zip(self.lb, self.ub)]
        chromosome.fitness = self.calculate_fitness(chromosome)
        return chromosome

# %% 快速非支配排序
    def fast_nondominated_sort(self,
                               population):
        front_set = [[]]
        for master in population:
            master.dominated_counter = 0
            master.dominat_solutions = []
            for slave in population:
                if master.dominates(slave):
                    master.dominat_solutions.append(slave)
                elif slave.dominates(master):
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
    def create_children(self,
                        parent):
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
    def tournament_selection(self,
                             parent):
        participants = random.sample(parent,
                                     k=self.tour_k)
        best = None
        for participant in participants:
            r = random.random()
            if (best is None) or (self.crowding_operator(participant, best) and r <= self.tour_prob):
                best = participant
        return best

    def crowding_operator(self, participant, best):
        if (participant.rank < best.rank) or (participant.rank == best.rank and participant.crowding_distance > best.crowding_distance):
            return True
        else:
            return False

# %% 交配
    def crossover(self,
                  p1,
                  p2):
        c1 = self.create_chromosome()
        c2 = self.create_chromosome()
        for gene_idx in range(self.D):
            beta = self.get_beta()
            gene1 = (p1.feature[gene_idx] + p2.feature[gene_idx]) / 2
            gene2 = abs((p1.feature[gene_idx] - p2.feature[gene_idx]) / 2)
            c1.feature[gene_idx] = gene1 + beta * gene2
            c2.feature[gene_idx] = gene1 - beta * gene2
        return c1, c2

    def get_beta(self):
        u = random.random()
        if u <= 0.5:
            beta = (2 * u) ** (1 / (self.cross_param + 1))
        else:
            beta = (2 * (1 - u)) ** (-1 / (self.cross_param + 1))
        return beta

# %% 突變
    def mutation(self,
                 c1):
        for gene_idx in range(self.D):
            u, delta = self.get_delta()

            if u < 0.5:
                c1.feature[gene_idx] += delta * (c1.feature[gene_idx] - self.lb[gene_idx])
            else:
                c1.feature[gene_idx] += delta * (self.ub[gene_idx] - c1.feature[gene_idx])

            if c1.feature[gene_idx] > self.ub[gene_idx]:
                c1.feature[gene_idx] = self.ub[gene_idx]
            if c1.feature[gene_idx] < self.lb[gene_idx]:
                c1.feature[gene_idx] = self.lb[gene_idx]
        return c1

    def get_delta(self):
        u = random.random()
        if u < 0.5:
            delta = (2 * u) ** (1 / (self.mutation_param + 1)) - 1
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (self.mutation_param + 1))
        return u, delta

# %% 取得最佳解
    def get_gbest(self,
                  front):
        self.X_gbest = np.array([chromosome.feature for chromosome in front])
        self.F_gbest = np.array([chromosome.fitness for chromosome in front])

# %% 畫圖
    def plot(self,
             title=None,
             xlabel=None,
             ylabel=None):
        if self.F_gbest is not None:
            plt.figure()
            plt.title(title)
            plt.scatter(self.F_gbest[:, 0], self.F_gbest[:, 1])
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid()
