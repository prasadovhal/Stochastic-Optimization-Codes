#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sept 24 16:17:32 2018

@author: prasad
@Roll No. : CMS1731
"""
"""Genetic Algorithm using Real values for population"""

import numpy as np

def ObjFunction(x1,x2,x3):
    f = (x3 + 2) * x1**2 * x2
    return f
    
def inequalityConstraints(x1,x2,x3):
    g1 = 1 - ((x2**3 * x3) / (71785. * x1**4))
    g2 = 1 - ((140.45 *  x1) / (x2**2 * x3))
    g3 = ((x1 + x2)/1.5) - 1
    g4 = ((4*x2**2 - x1*x2) / (12566. * (x1**3 * x2 - x1**4))) + (1 / (5108. * x1**2)) - 1
    if g1 == np.inf : g1 = 1000.
    if g2 == np.inf : g2 = 1000.
    if g3 == np.inf : g3 = 1000.
    if g4 == np.inf : g4 = 1000.
    return g1,g2,g3,g4

def FindFitness(x,C):
    g1 , g2 , g3 , g4 = inequalityConstraints(x[0],x[1],x[2])
    fitness = ObjFunction(x[0],x[1],x[2]) + C * sum([(max(0,g1))**beta , (max(0,g2))**beta, (max(0,g3))**beta ,(max(0,g4))**beta])
    return fitness

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

"Simple Crossover"
def SimpleCrossover(newSolution,PopulationSize,NumOfVar):
    CrossOveredExamples = []
    while True:
        splitJunction = np.random.randint(NumOfVar-1)
        p1 = newSolution[np.random.randint(PopulationSize)]
        p2 = newSolution[np.random.randint(PopulationSize)] 
        CrossOveredExamples.append(np.append(p1[:splitJunction],p2[splitJunction:]))
        CrossOveredExamples.append(np.append(p1[splitJunction:],p2[:splitJunction]))
        if len(CrossOveredExamples) == PopulationSize:
            break
    return CrossOveredExamples

"Whole Arithmetic Crossover"
def WholeArithmeticCrossover(newSolution,PopulationSize,NumOfVar):
    CrossOveredExamples = []
    while True:
        p1 = newSolution[np.random.randint(PopulationSize)]
        p2 = newSolution[np.random.randint(PopulationSize)] 
        a = alpha * p1 + (1 - alpha) * p2
        b = alpha * p2 + (1 - alpha) * p1
        CrossOveredExamples.append(a)
        CrossOveredExamples.append(b)
        if len(CrossOveredExamples) == PopulationSize:
            break
    return CrossOveredExamples

"Local Arithmetic Crossover"
def LocalArithmeticCrossover(newSolution,PopulationSize,NumOfVar):
    CrossOveredExamples = []
    while True:
        alpha = np.random.uniform()
        p1 = newSolution[np.random.randint(PopulationSize)]
        p2 = newSolution[np.random.randint(PopulationSize)] 
        a = alpha * p1 + (1 - alpha) * p2
        b = alpha * p2 + (1 - alpha) * p1
        CrossOveredExamples.append(a)
        CrossOveredExamples.append(b)
        if len(CrossOveredExamples) == PopulationSize:
            break
    return CrossOveredExamples

"BLX-alpha Crossover"
def BLXalphaCrossover(newSolution,PopulationSize,NumOfVar):
    CrossOveredExamples = []
    while True:
        alpha = np.random.uniform()
        p1 = newSolution[np.random.randint(PopulationSize)]
        p2 = newSolution[np.random.randint(PopulationSize)] 
        tplist = []
        for i in range(len(p1)):
            hmin = min([p1[i],p2[i]])
            hmax = max([p1[i],p2[i]])
            I = hmax - hmin
            a = np.random.uniform(hmin - I * alpha , hmax + I * alpha)
            tplist.append(a)
        CrossOveredExamples.append(tplist)
        if len(CrossOveredExamples) == PopulationSize:
            break
    return CrossOveredExamples

"Uniform Mutation"
def UniformMutation(CrossOveredExamples,mutationProbability,k,Range):
    mutatePopulation = []
    for j in range(len(CrossOveredExamples)):
        mutationExample = CrossOveredExamples[j]
        if np.random.uniform(0 , mutationProbability + 0.01) <= mutationProbability:
            mutationExample[np.random.choice(len(mutationExample))] = np.random.uniform(Range[0],Range[1])
        mutatePopulation.append(mutationExample)
    return mutatePopulation


def delta(k , y):
    b = 0.5
    gm = 3
    return y * (1- (np.random.uniform())**(1-(k / gm)))**b

"Non Uniform Mutation"
def NonUniformMutation(CrossOveredExamples,mutationProbability,k,Range):
    mutatePopulation = []
    for j in range(len(CrossOveredExamples)):
        tau = np.random.uniform()
        randomEg = CrossOveredExamples[np.random.randint(len(CrossOveredExamples))]
        if np.random.uniform(0,mutationProbability + 0.01) <= mutationProbability:
            geneToReplace = np.random.randint(len(randomEg))
            if tau >= 0.5:
                randomEg[geneToReplace] = randomEg[geneToReplace] + delta(k , Range[1] - randomEg[geneToReplace])
            else:
                randomEg[geneToReplace] = randomEg[geneToReplace] - delta(k , randomEg[geneToReplace] - Range[0])
        
        mutatePopulation.append(randomEg)
        if len(mutatePopulation) == len(CrossOveredExamples):
            break
    
    return mutatePopulation


Range = (0,5)
PopulationSize = 1000
population = []
NumOfVar = 3
crossoverProbability = 0.7
mutationProbability = 1 / PopulationSize
generation = 1000
beta = 1
C = 0.1
alpha = np.random.uniform()

for i in np.arange(PopulationSize):
    population.append(np.random.uniform(low = Range[0],high = Range[1],size = NumOfVar))

for k in np.arange(generation):
    print("k",k)
    "Find fitness"
    fitness = []
    for i in np.arange(PopulationSize):
        fitness.append(FindFitness(population[i],C))
    
    "Tournament Seletion"
    newSolution = TournamentSelection(fitness,PopulationSize)
    
    "Crossover"
    CrossOveredExamples = FlatCrossover(newSolution,PopulationSize,NumOfVar)
    
    "Mutation"
    mutatePopulation = UniformMutation(CrossOveredExamples,mutationProbability,k,Range)
    
    population = CrossOveredExamples

fitness = []
for i in np.arange(PopulationSize):
    fitness.append(FindFitness(population[i],C))

newSolution = TournamentSelection(fitness,PopulationSize)
print(newSolution[np.argmin(fitness)])

x1,x2,x3 = newSolution[np.argmin(fitness)]
print(inequalityConstraints(x1,x2,x3))

"""[0.0760559928876441, 1.198813417710683, 1.3981771260604874]
(-0.002881251495772119, -4.316059102698425, -0.15008705960111524, -0.05469839523174491)"""