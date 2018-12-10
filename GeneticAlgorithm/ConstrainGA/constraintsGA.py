#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 22:10:35 2018

@author: prasad
@Roll No : CMS1731
"""

"Genetic Algorithm with constrains"
import numpy as np

def BinToDec(x):
    summation = 0
    for i in np.arange(len(x)):
        summation = summation + x[i] * 2**(len(x)-1-i)
    return summation

"Minimize Objective function"
def ObjFunction(x1,x2,x3):
    f = (x3 + 2) * x1**2 * x2
    return f
    
def inequalityConstraints(x1,x2,x3):
    g1 = 1 - ((x2**3 * x3) / (71785. * x1**4))
    g2 = 1 - ((140.45 *  x1) / (x2**2 * x3))
    g3 = ((x1 + x2)/1.5) - 1
    g4 = ((4*x2**2 - x1*x2) / (12566. * (x1**3 * x2 - x1**4))) + (1 / (5108. * x1**2)) - 1
    if g1 == np.inf : g1 = 10000.
    if g2 == np.inf : g2 = 10000.
    if g3 == np.inf : g3 = 10000.
    if g4 == np.inf : g4 = 10000.
    return g1,g2,g3,g4

def FindFitness(x,C):
    g1 , g2 , g3 , g4 = inequalityConstraints(x[0],x[1],x[2])
    fitness = ObjFunction(x[0],x[1],x[2]) + C * sum([(max(0,g1))**beta , (max(0,g2))**beta, (max(0,g3))**beta ,(max(0,g4))**beta])
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
        splitJunction = np.random.randint(3*bit-1)
        p1 = newSolution[np.random.randint(len(newSolution))]
        p2 = newSolution[np.random.randint(len(newSolution))] 
        if splitJunction > 3*bit:
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
        for i in np.arange(3*bit):
            if np.random.uniform(0,(mutationProbability)) < mutationProbability:
                flip.append(abs(mutationExample[i] - 1))
            else:
                flip.append(mutationExample[i])
        mutatePopulation.append(np.array(flip))
    return mutatePopulation

x1Range = (0,5)
x2Range = (0,5)
x3Range = (0,20)

bit = 10
PopulationSize = 100
population = []
C = 0.1
beta = 1
generations = 5000

for i in np.arange(PopulationSize):
    population.append(np.random.randint(low = 0,high = 2,size = 3 * bit))

crossoverProbability = 0.7
mutationProbability = 0.3 #1 / len(population)
k = 0
for i in range(generations):
    k += 1
    print(k)
    "decode the population "
    x = []
    for i in np.arange(PopulationSize):
        a =   list(population[i][0:bit])
        b =   list(population[i][bit:2*bit])
        c =   list(population[i][2*bit:])
        x1 = x1Range[0] + ((x1Range[1] - x1Range[0]) / (2**bit - 1)) * BinToDec(a)
        x2 = x2Range[0] + ((x2Range[1] - x2Range[0]) / (2**bit - 1)) * BinToDec(b)
        x3 = x3Range[0] + ((x3Range[1] - x3Range[0]) / (2**bit - 1)) * BinToDec(c)
        x.append([x1,x2,x3])
        
    "finding fitness"
    fitness = []
    for i in range(PopulationSize):
        fitness.append(FindFitness(x[i],C))
    
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

a =   list(population[i][0:bit])
b =   list(population[i][bit:2*bit])
c =   list(population[i][2*bit:])
x1 = x1Range[0] + ((x1Range[1] - x1Range[0]) / (2**bit - 1)) * BinToDec(a)
x2 = x2Range[0] + ((x2Range[1] - x2Range[0]) / (2**bit - 1)) * BinToDec(b)
x3 = x3Range[0] + ((x3Range[1] - x3Range[0]) / (2**bit - 1)) * BinToDec(c)

print(x1,x2,x3)
x = [x1,x2,x3]
print(FindFitness(x,C))
print(inequalityConstraints(x1,x2,x3))
