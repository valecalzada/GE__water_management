import math

import numpy as np
from random import randint, uniform

class DE:
    def __init__(self, population_size, dimension, Cr, F, lower_limit, upper_limit, fitness_function, criterion, cut_points, print_evolution):
        self.population_size = population_size
        self.dimension = dimension
        self.Cr = Cr
        self.F = F
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.fitness_function = fitness_function
        self.population = []
        self.flipped_population = [] #For JSIGMA in Maximization
        self.selected = []
        self.mu = []
        self.nu = []
        self.t = 0
        self.best_evolved = []
        self.new_population = []
        self.best = None
        self.print_evolution = print_evolution
        self.function_calls = 0
        self.criterion = criterion
        self.cut_points = cut_points
        self.cut_points_fit_values = np.zeros(len(cut_points))

    def initialize_population(self):
        for i in range(self.population_size):
            individual = Individual(self.fitness_function)
            individual.initialize(self.lower_limit, self.upper_limit, self.dimension)
            individual.evaluate()
            self.population.append(individual)
        # reverse= True, to order from highest to lowest
        self.population.sort(key=lambda f: f.fitness_value, reverse=False)

    def mutation(self):
        pass

    def statistical_inference(self):
        pass

    def crossover_with_fix_bounds(self, ind, w):
        cross = []
        for i in range(0, self.dimension):
            if uniform(0, 1) <= self.Cr:
                cross.append(w[i])
            else:
                cross.append(ind[i])
        crossed = Individual(self.fitness_function)
        crossed.chromosome = self.fix_bounds(np.array(cross))
        crossed.evaluate()
        return crossed

    def crossover(self, ind, w):
        cross = []
        for i in range(0, self.dimension):
            if uniform(0, 1) <= self.Cr:
                cross.append(w[i])
            else:
                cross.append(ind[i])
        crossed = Individual(self.fitness_function)
        crossed.chromosome = np.array(cross)
        crossed.evaluate()
        return crossed

    def selection(self, crossed, ind):
        if crossed.fitness_value <= ind.fitness_value:
            return crossed
        else:
            return ind

    def replace_population(self):
        self.population.clear()
        for ind in self.new_population:
            self.population.append(ind)
        self.population.sort(key=lambda f: f.fitness_value, reverse=False)

    def flip_to_maximization(self):
        self.flipped_population = []
        for ind in self.population:
            ind.fitness_value = 1E16 - ind.fitness_value
            self.flipped_population.append(ind)

    def classic_selection_method(self):
        self.selected = []
        for i in range(0, int(self.population_size * 0.20)):
            self.selected.append(self.flipped_population[i])

    def selection_method(self):
        if self.t == 0:
            self.teta = self.flipped_population[-1]
        else:
            population_aux = [ind for ind in self.flipped_population if ind.fitness_value >= self.teta.fitness_value]
            g_min = population_aux[-1]
            g_mid = self.flipped_population[round(self.population_size / 2) - 1]

            if g_min.fitness_value >= g_mid.fitness_value:
                self.teta = g_min
            else:
                self.teta = g_mid
        self.selected = [ind for ind in self.flipped_population if ind.fitness_value >= self.teta.fitness_value]

    def fix_bounds(self, chromosome):
        min_value = min(chromosome)
        max_value = max(chromosome)
        return np.array([(value - min_value) / (max_value - min_value) * 255 for value in chromosome])

    def evolution(self, iterations):
        self.initialize_population()
        self.best = self.population[0]

        if self.print_evolution:
            print("Here comes the evolution...")
            print("0", self.best.fitness_value)

        for i in range(iterations):
            self.function_calls += 1
            self.new_population = []

            self.statistical_inference()
            for ind in self.population:
                mutant_chromosome = self.mutation()
                trial = self.crossover_with_fix_bounds(ind.chromosome, mutant_chromosome)
                selected = self.selection(trial, ind)
                self.new_population.append(selected)

            self.replace_population()
            self.best_evolved.append(self.population[0])
            self.best = self.population[0]

            if self.print_evolution:
                print((i + 1), self.best.fitness_value)  # Print Function Calls

            for c in range(0, len(self.cut_points)):
                if i * self.population_size <= self.cut_points[c]:
                    self.cut_points_fit_values[c] = self.best.fitness_value

            if self.population[0].fitness_value <= self.criterion:
                break

        return self.best_evolved, (self.function_calls * self.population_size), self.cut_points_fit_values


class Individual:
    def __init__(self, fitness_function):
        self.fitness_function = fitness_function

    def initialize(self, lim_inf, lim_sup, dimension):
        self.chromosome = np.random.uniform(lim_inf, lim_sup, size=dimension)

    def evaluate(self):
        self.fitness_value = self.fitness_function.evaluate(self.chromosome)


class Rand_Bin(DE):
    def mutation(self):
        E1 = self.population[randint(0, self.population_size - 1)]
        E2 = self.population[randint(0, self.population_size - 1)]
        E3 = self.population[randint(0, self.population_size - 1)]
        return E1.chromosome + self.F * (E2.chromosome - E3.chromosome)


class Rand_Best(DE):
    def mutation(self):
        E1 = self.population[0]  # Best individual
        E2 = self.population[randint(0, self.population_size - 1)]
        E3 = self.population[randint(0, self.population_size - 1)]
        return E1.chromosome + self.F * (E2.chromosome - E3.chromosome)


class Gaussian(DE):
    def statistical_inference(self):
        self.flip_to_maximization()
        self.selection_method()

        self.selected = [ind.chromosome for ind in self.selected]
        self.selected = np.array(self.selected).transpose()
        self.mu = [np.mean(dimensions) for dimensions in self.selected]
        #self.nu = [np.var(dimensions) for dimensions in self.selected]

    def mutation(self):
        E1 = self.mu
        E2 = self.population[randint(0, self.population_size - 1)]
        E3 = self.population[randint(0, self.population_size - 1)]
        return E1 + self.F * (E2.chromosome - E3.chromosome)


class JSIGMA(DE):
    def statistical_inference(self):
        self.flip_to_maximization()
        self.selection_method()
        self.mu = 0
        self.nu = 0
        delta = 0.001
        beta = 1 / self.selected[0].fitness_value

        exp_beta_gx = [math.exp(beta * xi.fitness_value) for xi in self.selected]
        Z = 1 / (sum(exp_beta_gx) * delta)

        # Compute mu
        for xi in self.selected:
            self.mu += (1 / Z) * (math.exp(beta * xi.fitness_value) * xi.chromosome) * delta

        # Compute nu
        #for xi in self.selected:
        #    self.nu += (1 / Z) * (math.exp(beta * xi.fitness_value) * ((xi.chromosome - self.mu) ** 2)) * delta

    def mutation(self):
        E1 = self.mu
        E2 = self.population[randint(0, self.population_size - 1)]
        E3 = self.population[randint(0, self.population_size - 1)]
        return E1 + self.F * (E2.chromosome - E3.chromosome)


def print_population(population):
    for individual in population:
        print(individual.chromosome, individual.fitness_value)