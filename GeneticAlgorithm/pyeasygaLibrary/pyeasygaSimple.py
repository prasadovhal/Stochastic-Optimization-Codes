#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:04:46 2018

@author: prasad
"""

from pyeasyga import pyeasyga

data = [('pear',50),('apple',35),('banana',40)]
ga = pyeasyga.GeneticAlgorithm(data)

def fitness(individual,data):
    fitness = 0
    if individual.count(1) == 2:
        for (selected,(fruit,profit)) in zip(individual,data):
            if selected:
                fitness += profit
    return fitness

ga.fitness_function = fitness
ga.run()
print(ga.best_individual())


