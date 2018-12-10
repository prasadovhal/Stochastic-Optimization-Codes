#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:17:32 2018

@author: prasad
@Roll No. : CMS1731
"""
"""Genetic Algorithm"""
import numpy as np

"Binary to Decimal conversion"
def BinToDec(x):
    summation = 0
    for i in np.arange(len(x)):
        summation = summation + x[i] * 2**(len(x)-1-i)
    return summation

"Objective functions"
def RosenBrock(x1,x2):
   a = 1.0
   b = 100.0
   return ((a - x1)**2 + (b * (x2 - x1**2)**2))

def DeJong(x,y):
    return (x**2 + y**2)

"Decode Population"
def decodePolpulation(population,Range,bit):
    x = []
    for i in np.arange(PopulationSize):
        a =   list(population[i][0:int(bit/2)])
        b =   list(population[i][int(bit/2):]) 
        x1 = Range[0] + ((Range[1] - Range[0]) / (2**bit - 1)) * BinToDec(a)
        x2 = Range[0] + ((Range[1] - Range[0]) / (2**bit - 1)) * BinToDec(b)
        x.append([x1,x2])
    return x

"Selection method"
def TournamentSelection(fitness,Range):
    fitterSolutions = []
    while True:
        p1 = fitness[np.random.randint(Range[1])]
        p2 = fitness[np.random.randint(Range[1])]
   
        if p1 <= p2:
            fitterSolutions.append(p1)
        
        if len(fitterSolutions) == len(fitness):
            break
    
    fitterSolutionsIndex = []
    for i in np.arange(len(fitterSolutions)):
        fitterSolutionsIndex.append(np.where(fitterSolutions[i] == fitness)[0][0])
    
    newSolution = []
    for i in np.arange(len(fitterSolutionsIndex)):
        newSolution.append(population[fitterSolutionsIndex[i]])
        
    return newSolution

"Crossover function"
def Crossover(newSolution,Range,bit,crossoverProbability):
    CrossOveredExamples = []
    while True:        
        splitJunction = np.random.randint(bit-1)
        p1 = newSolution[np.random.randint(Range[1])]
        p2 = newSolution[np.random.randint(Range[1])]
    
        if splitJunction >= bit:
            CrossOveredExamples.append(np.append(p1[:splitJunction],p2[splitJunction:]))
        else:  
            CrossOveredExamples.append(np.append(p1[splitJunction:],p2[:splitJunction]))
       
        if len(CrossOveredExamples) == len(newSolution):
            break
    return CrossOveredExamples

"Mutation"
def Mutation(CrossOveredExamples,bit,mutationProbability):
    mutatePopulation = []
    for j in np.arange(len(CrossOveredExamples)):
        mutationExample = CrossOveredExamples[j]
        flip = []
        for i in np.arange(bit):
            if np.random.uniform(0,(mutationProbability+0.01)) < mutationProbability:
                flip.append(abs(mutationExample[i] - 1))
            else:
                flip.append(mutationExample[i])
        mutatePopulation.append(np.array(flip))
    return mutatePopulation

bit = 30
Range = (-2,2)
PopulationSize = 500
generation = 1000
crossoverProbability = 0.7
mutationProbability = 1 / PopulationSize
population = []

for i in np.arange(PopulationSize):
    population.append(np.random.randint(low = 0,high = 2,size = bit))

for k in np.arange(generation):
    print("k",k)
    "decode the population "
    x = decodePolpulation(population,Range,bit)
    
    "Find fitness"
    fitness = []
    for i in np.arange(PopulationSize):
        fitness.append(RosenBrock(x[i][0],x[i][1]))
    
    "Tournament Seletion"
    newSolution = TournamentSelection(fitness,Range)
    
    "Crossover"
    CrossOveredExamples = Crossover(newSolution,Range,bit,crossoverProbability)
    
    "Mutation"
    mutatePopulation = Mutation(CrossOveredExamples,bit,mutationProbability)
    
    population = mutatePopulation

x = decodePolpulation(population,Range,bit)
fitness = []
for i in np.arange(PopulationSize):
    fitness.append(RosenBrock(x[i][0],x[i][1]))

x1,x2 = x[np.argmin(fitness)]
print(x1,x2)
print(RosenBrock(x1,x2))
