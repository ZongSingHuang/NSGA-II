from nsga2.utils import NSGA2Utils
from nsga2.population import Population

class Evolution:

    def __init__(self, problem, num_of_generations=1000, num_of_individuals=100, num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5):
        self.utils = NSGA2Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob, crossover_param, mutation_param)
        self.population = None
        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals

    def evolve(self):
        # 建立 P 條染色體作為父代，並計算各自的適應值
        self.population = self.utils.create_initial_population()
        # 取得父代染色體的排名，並且分群
        self.utils.fast_nondominated_sort(self.population)
        # 計算父代染色體各群ㄋ的擁擠度
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        # 建立子代 : 選擇 -> 交配 -> 突變
        children = self.utils.create_children(self.population)
        # 用來放 父代 + 子代 用的
        returned_population = None
        # 開始迭代
        for i in range(self.num_of_generations):
            # 父代與子代合併
            self.population.extend(children)
            # 取得父代 + 子代染色體的排名，並且分群
            self.utils.fast_nondominated_sort(self.population)
            # 建立空的容器
            new_population = Population()
            # 菁英策略，逐批取的群，同時計算擁擠度，直到把容器塞滿或者快滿
            front_num = 0
            while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals:
                print(f'現在 new_population 的長度 {len(new_population)}')
                print(f'第 {front_num} 群 的長度 {len(self.population.fronts[front_num])}')
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1
                print(f'更新後 new_population 的長度 {len(new_population)}')
            # 計算 父代 + 子代 front_num + 1 群的擁擠度
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            # 對 父代 + 子代 front_num + 1 群的染色體依擁擠度作排序
            self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
            # 若容器還沒滿，則用父代 + 子代 front_num + 1 群的染色體充數
            new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals-len(new_population)])
            # 把父代 + 子代備份起來
            returned_population = self.population
            # 容器作為新父代
            self.population = new_population
            # 取得新父代染色體的排名，並且分群
            self.utils.fast_nondominated_sort(self.population)
            # 計算新父代染色體各群的擁擠度
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            # 建立子代 : 選擇 -> 交配 -> 突變
            children = self.utils.create_children(self.population)
        return returned_population.fronts[0]
