#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sept 24 16:17:32 2018

@author: prasad
@Roll No. : CMS1731
"""
"""Genetic Algorithm using Real values for population"""

import numpy as np

def RosenBrock(x1,x2):
   return ((1.0 - x1)**2 + (100.0 * (x2 - x1**2)**2))

def TournamentSelection(fitness,PopulationSize):
    fitterSolutions = []
    while True:
        p1 = fitness[np.random.randint(PopulationSize)]
        p2 = fitness[np.random.randint(PopulationSize)]
        if p1 <= p2:
            fitterSolutions.append(p1)
        if len(fitterSolutions) == PopulationSize:
            break
    
    fitterSolutionsIndex = []
    for i in range(PopulationSize):
        for j in range(PopulationSize):
            if fitterSolutions[i] == fitness[j] :
                fitterSolutionsIndex.append(j)        
    newSolution = []
    for i in np.arange(len(fitterSolutionsIndex)):
        newSolution.append(population[fitterSolutionsIndex[i]])
    
    return newSolution

"Flat Crossover"
def FlatCrossover(newSolution,PopulationSize,NumOfVar):
    CrossOveredExamples = []
    while True:
        p1 = newSolution[np.random.randint(PopulationSize)]
        p2 = newSolution[np.random.randint(PopulationSize)] 
        tplist = []
        for i in range(NumOfVar):
            tplist.append(np.random.uniform(p1[i],p2[i]))
        CrossOveredExamples.append(tplist)
        if len(CrossOveredExamples) == PopulationSize:
            break
    return CrossOveredExamples

def UniformMutation(CrossOveredExamples,mutationProbability):
    mutatePopulation = []
    for j in range(len(CrossOveredExamples)):
        mutationExample = CrossOveredExamples[j]
        if np.random.uniform(0,mutationProbability+0.01) <= mutationProbability:
            mutationExample[np.random.choice(len(mutationExample))] = np.random.uniform(Range[0],Range[1])
        mutatePopulation.append(mutationExample)
    return mutatePopulation


Range = (-5,5)
PopulationSize = 1000
population = []
NumOfVar = 2
crossoverProbability = 0.7
mutationProbability = 1 / PopulationSize
generation = 1000

for i in np.arange(PopulationSize):
    population.append(np.random.uniform(low = Range[0],high = Range[1],size = NumOfVar))

for k in np.arange(generation):
    print("k",k)
    "Find fitness"
    fitness = []
    for i in np.arange(PopulationSize):
        fitness.append(RosenBrock(population[i][0],population[i][1]))
    
    "Tournament Seletion"
    newSolution = TournamentSelection(fitness,PopulationSize)
    
    "Crossover"
    CrossOveredExamples = FlatCrossover(newSolution,PopulationSize,NumOfVar)
    
    "Mutation"
    mutatePopulation = UniformMutation(CrossOveredExamples,mutationProbability)
    
    population = CrossOveredExamples

fitness = []
for i in np.arange(PopulationSize):
    fitness.append(RosenBrock(population[i][0],population[i][1]))

newSolution = TournamentSelection(fitness,PopulationSize)
print(newSolution[np.argmin(fitness)])




""" fitness = []
    for i in range(len(newSolution)):
        fitness.append(RosenBrock(population[i][0],population[i][1]))
    EliteSolutions = list(np.array(newSolution)[(list(np.argsort(fitness)[0:int((1-crossoverProbability) * PopulationSize)]))])"""