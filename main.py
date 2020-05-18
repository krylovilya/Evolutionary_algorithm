import random
import sys
import time
import warnings
from collections import OrderedDict

import mainwindow
import numpy as np
from PyQt5 import QtWidgets


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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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


class Population:
    def __init__(self, n):
        self.n = n
        self.vector = []

    def gen_start_pop(self):
        for i in range(self.n):
            self.vector.append(Individual())

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
    def __init__(self, crossover_chance, mutation_chance, stop_criterion, print_to_console, population_size):
        self.pop = Population(population_size)
        self.population_size = population_size
        self.pop.gen_start_pop()
        self.stop_criterion = stop_criterion
        self.crossover_chance = crossover_chance
        self.mutation_chance = mutation_chance
        self.iter = 0
        self.max_fitness = None
        self.unchanged = 0
        self.time = None
        self.print_to_console = print_to_console
        self.data_for_graph1 = []
        self.data_for_graph2 = []

    def rank_selection(self):
        old_population = self.pop
        fitness = OrderedDict(sorted({el.fitness_function(): [el, 0] for el in old_population.vector}.items()))
        sum_ranks = 0
        for i, key in enumerate(fitness):
            fitness[key][1] = i + 1
            sum_ranks += i + 1
        new_fitness = OrderedDict({fitness[key][0]: fitness[key][1] for key in fitness})
        n = [round(2 * ((new_fitness[key] - 1) / (self.pop.n - 1))) for key in new_fitness]
        new_population = Population(self.population_size)
        fitness_iter = reversed(new_fitness)
        for _n in list(reversed(n)):
            current_obj = next(fitness_iter)
            for j in range(_n):
                new_population.vector.append(current_obj)
            if _n == 0 and len(new_population.vector) < self.pop.n:
                new_population.vector.append(current_obj)
        while len(new_population.vector) < self.pop.n:
            new_population.vector.append(random.choice(new_population.vector))
        self.pop = new_population

    def single_point_crossover(self):
        old_population = self.pop
        old_population.vector = random.sample(old_population.vector, len(old_population.vector))
        new_population = Population(self.population_size)
        pairs = list(zip(old_population.vector[:int(self.pop.n / 2)], old_population.vector[int(self.pop.n / 2):]))
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
        rnd = [random.random() for i in range(self.pop.n)]
        for _rnd, el in zip(rnd, self.pop.vector):
            if _rnd <= self.mutation_chance:
                el.mutation()

    def stop_function(self):
        if self.stop_criterion[0]:
            condition1 = self.unchanged > self.stop_criterion[0] - 1
        else:
            condition1 = False
        if self.stop_criterion[1]:
            condition2 = np.isinf(self.pop.get_max())
        else:
            condition2 = False
        if self.stop_criterion[2]:
            condition3 = time.time() - self.time > self.stop_criterion[2]
        else:
            condition3 = False
        return condition1 or condition2 or condition3

    def start(self):
        self.time = time.time()
        self.max_fitness = self.pop.get_max()
        result = None
        while True:
            if self.print_to_console:
                print(self.iter, self.unchanged, self.pop.get_statistics())
            stop = self.stop_function()
            if not stop:
                self.iter += 1
                self.rank_selection()
                self.single_point_crossover()
                self.mutation()
                max_value = self.pop.get_max()
                if max_value > self.max_fitness or \
                        not np.isinf(max_value) and np.isinf(self.max_fitness):
                    self.max_fitness = max_value
                    self.unchanged = 0
                else:
                    self.unchanged += 1
                _mean = self.pop.get_mean()
                _max = self.pop.get_max()
                if np.isinf(_mean):
                    _mean = 1e10
                if np.isinf(_max):
                    _max = 1e10
                self.data_for_graph1.append(_mean)
                self.data_for_graph2.append(_max)
            else:
                result = self.pop.vector[np.argmax(map(Individual.fitness_function, self.pop.vector))]
                break
        return result.get_phenotype()


class MainApp(QtWidgets.QMainWindow, mainwindow.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.legend = None
        self.a = None
        self.graphicsView.setLabel('left', "Функция приспособленности ")
        self.graphicsView.setLabel('bottom', "Номер популяции")
        self.graphicsView.setLogMode(y=True)
        self.graphicsView.enableAutoRange('xy', True)

    def button_click(self):
        self.pushButton.setEnabled(False)
        crossover_chance = self.doubleSpinBox.value() / 100
        mutation_chance = self.doubleSpinBox_2.value() / 100
        population_size = self.spinBox_2.value()
        stop_criterion = []
        if self.checkBox.isChecked():
            stop_criterion.append(self.spinBox.value())
        else:
            stop_criterion.append(False)
        stop_criterion.append(self.checkBox_2.isChecked())
        if self.checkBox_3.isChecked():
            stop_criterion.append(self.doubleSpinBox_3.value())
        else:
            stop_criterion.append(False)
        if not any(stop_criterion):
            QtWidgets.QMessageBox.critical(self, "Ошибка", "Необходим хотя бы один критерий останова!")
            self.pushButton.setEnabled(True)
            return
        print_to_console = self.checkBox_4.isChecked()
        self.a = Algorithm(crossover_chance, mutation_chance, stop_criterion, print_to_console, population_size)
        result = self.a.start()
        if print_to_console:
            print("\n 1 значение: номер популяции\n",
                  "2 значение: количество поколений без улучшений\n",
                  "3 значение: индивид с минимальным значением функции приспособленности в популяции\n",
                  "4 значение: индивид со средним значением функции приспособленности в популяции\n",
                  "5 значение: индивид с максимальным значением функции приспособленности в популяции\n")
        self.label_7.setText(str(result)[1:-1])
        self.label_9.setText(str(self.a.iter))
        self.label_11.setText(str(round(time.time() - self.a.time, 4)) + " с.")
        self.pushButton.setEnabled(True)
        self.draw(self.checkBox_6.isChecked())

    def draw(self, symbol=False):
        symbol_dict2, symbol_dict1 = {}, {}
        if symbol:
            symbol_dict2 = {'symbol': "o", 'symbolBrush': (255, 0, 0), 'symbolSize': 10}
            symbol_dict1 = {'symbol': "o", 'symbolBrush': (0, 0, 255), 'symbolSize': 10}
        self.graphicsView.clear()
        if self.legend:
            self.legend.scene().removeItem(self.legend)
        y1 = self.a.data_for_graph1
        y2 = self.a.data_for_graph2
        self.legend = self.graphicsView.addLegend()
        self.graphicsView.plot(x=np.arange(self.a.iter), y=y2, pen=(255, 0, 0), name="макс. знач.", **symbol_dict2)
        self.graphicsView.plot(x=np.arange(self.a.iter), y=y1, pen=(0, 0, 255), name="ср. знач.", **symbol_dict1)

    def spinbox_changed(self, new_val):
        self.checkBox.setText(new_val.__str__() + " поколений без улучшений")

    def time_spinbox_changed(self, new_val):
        self.checkBox_3.setText(str(new_val) + " сек. процессорного времени")

    def change_scale(self, val):
        self.graphicsView.setLogMode(y=val)

    def change_symbol(self, val):
        if self.a:
            self.draw(val)


def main():
    print("launch Qt5...")
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
