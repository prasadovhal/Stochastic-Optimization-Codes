#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 00:11:42 2018

@author: prasad
@Roll Number: CMS1731
"""

import numpy as np
import matplotlib.pyplot as plt
#from pybrain.tools.nondominated import crowding_distance

"Minimize both objective function"
def obj1(x):
    return x**2

def obj2(x):
    return (x-2)**2

def BinToDec(x):
    summation = 0
    for i in np.arange(len(x)):
        summation = summation + x[i] * 2**(len(x)-1-i)
    return summation

def Decodepopulation(population,Range):
    x = []
    for i in np.arange(len(population)):
        a =   list(population[i])
        x1 = Range[0] + ((Range[1] - Range[0]) / (2**bit - 1)) * BinToDec(a)
        x.append(x1)
    return x

"Upate Pareto Set"
def Remove(duplicate): 
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 

def ParetoSetUpdate(ParetoSet , x_new):
    Set = []
    E1 = [obj1(x_new) , obj2(x_new)]
    for i in range(len(ParetoSet)):
        E2 = [obj1(ParetoSet[i]) , obj2(ParetoSet[i])]
        if E1[0] < E2[0] and E1[1] < E2[1]:
           Set.append(x_new)
        elif E1[0] > E2[0] and E1[1] > E2[1]:
            Set.append(ParetoSet[i])
        else:
            Set.append(x_new)
            Set.append(ParetoSet[i])
    return Remove(Set)

"Rank Examples Function"
def RankSolution(x):
    RankedSolution = []
    while len(x) > 1:
        ParetoSet = [x[0]]
        for i in range(1,len(x)):
            ParetoSet = ParetoSetUpdate(ParetoSet , x[i])
    
        RankedSolution.append(ParetoSet)
        for i in range(len(ParetoSet)):
            x.remove(ParetoSet[i])
    
    if len(x) == 1:
        RankedSolution.append(x)
    
    return RankedSolution

"Tournament Selection Function"
def TournamentSelection(RankedSolution,PopulationSize):
    FitterSolution = []
    while True:
        a = np.random.randint(len(RankedSolution))
        b = np.random.randint(len(RankedSolution))
        c = RankedSolution[a]
        d = RankedSolution[b]
        p1 = c[np.random.randint(len(c))]
        p2 = d[np.random.randint(len(d))]
        
        if a < b:
            FitterSolution.append(p1)
        else:
            FitterSolution.append(p2)
        
        if len(FitterSolution) == PopulationSize:
            break
        
    return FitterSolution

"Crossover function"
def Crossover(FitterSolution,bit):
    CrossOveredExamples = []
    while True:    
        splitJunction = np.random.randint(bit-1)
        p1 = FitterSolution[np.random.randint(len(FitterSolution))]
        p2 = FitterSolution[np.random.randint(len(FitterSolution))] 
        if splitJunction > bit:
            CrossOveredExamples.append(np.append(p1[:splitJunction],p2[splitJunction:]))
        else:  
            CrossOveredExamples.append(np.append(p1[splitJunction:],p2[:splitJunction]))
            
        if len(CrossOveredExamples) == len(FitterSolution):
            break
    return CrossOveredExamples

"Mutation function"
def Mutation(CrossOveredExamples,bit,mutationProbability):
    mutatePopulation = []
    for j in np.arange(len(CrossOveredExamples)):
        mutationExample = CrossOveredExamples[j]
        flip = []
        for i in np.arange(bit):
            if np.random.uniform(0,(mutationProbability)) < mutationProbability:
                flip.append(abs(mutationExample[i] - 1))
            else:
                flip.append(mutationExample[i])
        mutatePopulation.append(np.array(flip))
    return mutatePopulation


"Define Parameters"
bit = 10
Range = (-5,5)
PopulationSize = 300
crossoverProbability = 0.7
mutationProbability = 1 / PopulationSize
generation = 500
population = []

"Finding fitness"
for i in np.arange(PopulationSize):
    population.append(np.random.randint(low = 0,high = 2,size = bit))

for gen in range(generation):
    print(gen)
    "Decode Population"
    x = Decodepopulation(population , Range)
    
    "finding fitness"
    fitness = []
    for i in range(PopulationSize):
        fitness.append([obj1(x[i]) , obj2(x[i])])
    
    "Ranking Solutions"
    RankedSolution = RankSolution(x.copy())
    
    "Tournament Selection"
    FitterSolution = TournamentSelection(RankedSolution,PopulationSize)
    
    newSolution = []
    for i in range(PopulationSize):
        newSolution.append(population[np.where(FitterSolution[i] == x)[0][0]])
    
    "Crossover"
    CrossOveredExamples = Crossover(newSolution,bit)
   
    "Mutation"
    mutatePopulation = Mutation(CrossOveredExamples,bit,mutationProbability)
    
    population = mutatePopulation

x = Decodepopulation(population , Range)
fitness = []
for i in range(PopulationSize):
    fitness.append([obj1(x[i]) , obj2(x[i])])
    
plt.plot(x,np.array(fitness)[:,0])
plt.plot(x,np.array(fitness)[:,1])
plt.show()
