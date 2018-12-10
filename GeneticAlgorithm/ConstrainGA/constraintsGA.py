#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 22:10:35 2018

@author: prasad
"""

import numpy as np

def BinToDec(x):
    summation = 0
    for i in np.arange(len(x)):
        summation = summation + x[i] * 2**(len(x)-1-i)
    return summation

"Minimize Objective function"
def ObjFunction(x1,x2):
    f = (x1 - 10)**2 + (x2 - 20)**2
    return f
    
def inequalityConstraints(x1,x2):
    g1 = -(x1 - 5)**2 - (x2 - 5)**2 + 100   # <=0
    g2 = -(x1 - 6)**2 - (x2 - 5)**2 - 82.81 # <=0
    return g1,g2

def FindFitness(x,C):
    fitness = []
    for i in range(len(x)):
        tp = ObjFunction(x[i][0],x[i][1]) + C * sum(inequalityConstraints(x[i][0],x[i][1]))
        fitness.append(tp)
    return fitness

def TournamentSelection(fitness):
    fitterSolutions = []
    while True:
        p1 = fitness[np.random.randint(len(fitness))]
        p2 = fitness[np.random.randint(len(fitness))]        
        if p1 < p2:
            fitterSolutions.append(p1)        
        if len(fitterSolutions) == len(fitness):
            break
    return fitterSolutions

def Crossover(newSolution,bit):
    CrossOveredExamples = []
    while True:    
        splitJunction = np.random.randint(2*bit-1)
        p1 = newSolution[np.random.randint(len(newSolution))]
        p2 = newSolution[np.random.randint(len(newSolution))] 
        if splitJunction >= bit:
            CrossOveredExamples.append(np.append(p1[:splitJunction],p2[splitJunction:]))
        else:  
            CrossOveredExamples.append(np.append(p1[splitJunction:],p2[:splitJunction]))
            
        if len(CrossOveredExamples) == len(newSolution):
            break
    return CrossOveredExamples

def Mutation(CrossOveredExamples,bit,mutationProbability):
    mutatePopulation = []
    for j in np.arange(len(CrossOveredExamples)):
        mutationExample = CrossOveredExamples[j]
        flip = []
        for i in np.arange(2*bit):
            if np.random.uniform(0,(mutationProbability+0.01)) < mutationProbability:
                flip.append(abs(mutationExample[i] - 1))
            else:
                flip.append(mutationExample[i])
        mutatePopulation.append(np.array(flip))
    return mutatePopulation

x1Range = (13,100)
x2Range = (0,100)

bit = 20
PopulationSize = 1000
population = []
C = 0.001
alpha = 1
beta = 1
generations = 1000

for i in np.arange(PopulationSize):
    population.append(np.random.randint(low = 0,high = 2,size = 2 * bit))

crossoverProbability = 0.7
mutationProbability = 1 / len(population)
k = 0
for i in range(generations):
    k += 1
    print(k)
    "decode the population "
    x = []
    for i in np.arange(PopulationSize):
        a =   list(population[i][0:bit])
        b =   list(population[i][bit:]) 
        x1 = x1Range[0] + ((x1Range[1] - x1Range[0]) / (2**bit - 1)) * BinToDec(a)
        x2 = x2Range[0] + ((x2Range[1] - x2Range[0]) / (2**bit - 1)) * BinToDec(b)
        x.append([x1,x2])
        
    "finding fitness"
    fitness = FindFitness(x,C)
    "Tournament Seletion"
    fitterSolutions = TournamentSelection(fitness)

    fitterSolutionsIndex = []
    for i in np.arange(len(fitterSolutions)):
        fitterSolutionsIndex.append(np.where(fitterSolutions[i] == fitness)[0][0])
    
    newSolution = []
    for i in np.arange(len(fitterSolutionsIndex)):
        newSolution.append(population[fitterSolutionsIndex[i]])
        
    "Crossover"
    CrossOveredExamples = Crossover(newSolution,bit)
    "Mutation"
    mutatePopulation = Mutation(CrossOveredExamples,bit,mutationProbability)
    
    population = mutatePopulation

a = list(population[np.argmin(fitterSolutions)][0:bit])
b = list(population[np.argmin(fitterSolutions)][bit:])

x1 = x1Range[0] + ((x1Range[1] - x1Range[0]) / (2**bit - 1)) * BinToDec(a)
x2 = x2Range[0] + ((x2Range[1] - x2Range[0]) / (2**bit - 1)) * BinToDec(b)

print(x1,x2)
print(ObjFunction(x1,x2) + C * sum(inequalityConstraints(x1,x2)))

"""x1 = 18.828707531650096, x2 =  8.801182557280118
f = 202.99207192818614 """