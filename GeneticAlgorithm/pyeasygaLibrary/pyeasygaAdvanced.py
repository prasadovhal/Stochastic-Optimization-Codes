#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:09:51 2018

@author: prasad
"""

import numpy as np
import random
from pyeasyga import pyeasyga

data = [('pear',50),('apple',35),('banana',40)]

ga = pyeasyga.GeneticAlgorithm(data,
                               population_size = 10,generations = 20,
                               crossover_probability = 0.8,
                               mutation_probability = 0.05,
                               elitism = True,
                               maximise_fitness = True)

#ga = pyeasyga.GeneticAlgorithm(data,10,20,0.8,0.05,True,True)

def create_individual(data):
    return [np.random.randint(0,1) for _ in np.arange(len(data))]

ga.create_individual = create_individual

def crossover(parent_1, parent_2):
    crossover_index = random.randrange(1, len(parent_1))
    child_1 = parent_1[:crossover_index] + parent_2[crossover_index:]
    child_2 = parent_2[:crossover_index] + parent_1[crossover_index:]
    return child_1, child_2

ga.crossover_function = crossover

def mutate(individual):
    mutate_index = random.randrange(len(individual))
    if individual[mutate_index] == 0:
        individual[mutate_index] == 1
    else:
        individual[mutate_index] == 0

ga.mutate_function = mutate

def selection(population):
    return np.random.choice(population)

ga.selection_function = selection

def fitness (individual, data):
    fitness = 0
    if individual.count(1) == 2:
        for (selected, (fruit, profit)) in zip(individual, data):
            if selected:
                fitness += profit
    return fitness

ga.fitness_function = fitness
ga.run()

print(ga.best_individual())

for individual in ga.last_generation():
    print(individual)