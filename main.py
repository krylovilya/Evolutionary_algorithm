import random
from collections import OrderedDict
import time
import numpy as np


#   Индивид
class Individual:
    def __init__(self, x=None):
        if x is None:
            gene = []
            for i in range(24):
                gene.append(str(random.randrange(0, 2, 1)))
            x = ["".join(gene[:12]), "".join(gene[12:])]
        self.genotype = x
        self.phenotype = None

    def get_phenotype(self):
        if self.phenotype:
            return self.phenotype
        else:
            self.phenotype = [(int(self.genotype[i], 2) - 2048) / 1000.0 for i in range(2)]
            return self.phenotype

    def target_function(self):
        x = self.get_phenotype()
        xx = np.power(x, 2)
        return 100 * (xx[0] - xx[1]) * (xx[0] - xx[1]) + (1 - x[0]) * (1 - x[0])

    def fitness_function(self):
        return 1 / self.target_function()

    def mutation(self):
        pos = random.randint(0, 11)
        # first_gen_list = list(self.genotype[0])
        # second_gen_list = list(self.genotype[1])
        if self.genotype[0][pos] == '0':
            first_gen = self.genotype[0][:pos] + (self.genotype[0][pos:].replace('0', '1', 1))
        else:
            first_gen = self.genotype[0][:pos] + (self.genotype[0][pos:].replace('1', '0', 1))
        if self.genotype[1][pos] == '0':
            second_gen = self.genotype[1][:pos] + (self.genotype[1][pos:].replace('0', '1', 1))
        else:
            second_gen = self.genotype[1][:pos] + (self.genotype[1][pos:].replace('1', '0', 1))
        self.genotype = [first_gen, second_gen]
        self.phenotype = None

    def __str__(self):
        return self.get_phenotype().__str__()


#   Популяция
class Population:
    def __init__(self, n=100):
        self.n = n
        self.vector = []

    def gen_start_pop(self):
        for i in range(self.n):
            self.vector.append(Individual())

    def resize_pop(self, new_n):
        self.n = new_n

    def __len__(self):
        return self.vector.__len__()

    def get_min(self):
        return np.min(list(map(Individual.fitness_function, self.vector)))

    def get_mean(self):
        return np.mean(list(map(Individual.fitness_function, self.vector)))

    def get_max(self):
        return np.max(list(map(Individual.fitness_function, self.vector)))

    def get_statistics(self):
        return self.get_min(), self.get_mean(), self.get_max()


class Algorithm:
    def __init__(self):
        self.pop = Population()
        self.pop.gen_start_pop()
        self.crossover_chance = 0.6
        self.mutation_chance = 0.01
        self.iter = 0
        self.stop_criterion = 200
        self.max_fitness = None
        self.unchanged = 0
        self.time = None

    def rank_selection(self):
        old_population = self.pop
        fitness = OrderedDict(sorted({el.fitness_function(): [el, 0] for el in old_population.vector}.items()))
        sum_ranks = 0
        for i, key in enumerate(fitness):
            fitness[key][1] = i + 1
            sum_ranks += i + 1
        # fitness = OrderedDict({el.fitness_function(): i for i, el in enumerate(self.pop.vector)})
        # p = OrderedDict({fitness[key][0]: fitness[key][1] / sum_ranks for key in fitness})
        new_fitness = OrderedDict({fitness[key][0]: fitness[key][1] for key in fitness})
        # cumulate = 0
        # rnd = [random.random() for i in range(200)]

        n = [round(2 * ((new_fitness[key] - 1) / (100 - 1))) for key in new_fitness]
        new_population = Population()
        fitness_iter = reversed(new_fitness)
        for _n in list(reversed(n)):
            current_obj = next(fitness_iter)
            for j in range(_n):
                new_population.vector.append(current_obj)
            if _n == 0 and len(new_population.vector) < 100:
                new_population.vector.append(current_obj)
        while len(new_population.vector) < 100:
            new_population.vector.append(random.choice(new_population.vector))
        self.pop = new_population

    def single_point_crossover(self):
        old_population = self.pop
        old_population.vector = random.sample(old_population.vector, len(old_population.vector))
        new_population = Population()
        pairs = list(zip(old_population.vector[:50], old_population.vector[50:]))
        for pair in pairs:
            do_crossover = random.random() <= self.crossover_chance
            if do_crossover:
                break_point = random.randint(1, 12)
                first_parents = [pair[0].genotype[0], pair[1].genotype[0]]
                first_descendants = [
                    "{}{}".format(first_parents[0][:break_point], first_parents[1][break_point:]),
                    "{}{}".format(first_parents[1][:break_point], first_parents[0][break_point:])]
                second_parents = [pair[0].genotype[1], pair[1].genotype[1]]
                second_descendants = [
                    "{}{}".format(second_parents[0][:break_point], second_parents[1][break_point:]),
                    "{}{}".format(second_parents[1][:break_point], second_parents[0][break_point:])]
                new_population.vector.append(Individual([first_descendants[0], second_descendants[0]]))
                new_population.vector.append(Individual([first_descendants[1], second_descendants[1]]))
            else:
                new_population.vector.append(pair[0])
                new_population.vector.append(pair[1])
        self.pop = new_population

    def mutation(self):
        rnd = [random.random() for i in range(100)]
        for _rnd, el in zip(rnd, self.pop.vector):
            if _rnd <= self.mutation_chance:
                el.mutation()

    def stop_function(self):
        condition1 = self.unchanged > self.stop_criterion
        condition2 = self.pop.get_max() == np.inf
        condition3 = time.time() - self.time > 2
        return condition1 or condition2 or condition3

    def start(self):
        self.time = time.time()
        self.max_fitness = self.pop.get_max()
        result = None
        while True:
            print(self.iter, self.unchanged, self.pop.get_statistics())
            stop = self.stop_function()
            if not stop:
                self.iter += 1
                self.rank_selection()
                self.single_point_crossover()
                self.mutation()
                if self.pop.get_max() > self.max_fitness:
                    self.max_fitness = self.pop.get_max()
                    self.unchanged = 0
                else:
                    self.unchanged += 1
            else:
                # result = self.pop.vector[max(range(len(self.pop.vector)), key=self.pop.vector.__getitem__)]
                result = self.pop.vector[np.argmax(map(Individual.fitness_function, self.pop.vector))]
                break
        return result.get_phenotype()


if __name__ == "__main__":
    a = Algorithm()
    print(a.start())
    # print(len(a.pop))
    # for i in range(100):
    #    a.rank_selection()
    #    print(i, '\t',a.pop.get_statistics())
    #    a.single_point_crossover()
    #    print('\t', a.pop.get_statistics())
    #    a.mutation()
    #    print('\t', a.pop.get_statistics())
    # print(list(map(Individual.get_phenotype, pop.vector)))

#    pop = Population()
#    for i, el in enumerate(pop.vector):
#        print(i + 1, el.target_function(), el.fitness_function())
