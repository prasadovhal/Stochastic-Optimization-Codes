#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:17:32 2018

@author: prasad
@Roll No. : CMS1731
"""
"""Genetic Algorithm"""
import numpy as np
#import pandas as pd

def BinToDec(x):
    summation = 0
    for i in np.arange(len(x)):
        summation = summation + x[i] * 2**(len(x)-1-i)
    return summation

def RosenBrock(x1,x2):
   a = 1.0
   b = 100.0
   return ((a - x1)**2 + (b * (x2 - x1**2)**2))

def DeJong(x,y):
    return (x**2 + y**2)

bit = 18
Range = (-5,5)
PopulationSize = 10
population = []

for i in np.arange(PopulationSize):
    population.append(np.random.randint(low = 0,high = 2,size = bit))

crossoverProbability = 0.7
mutationProbability = 1 / len(population)

for k in np.arange(500):
    print("k",k)
    "decode the population "
    
    x = []
    for i in np.arange(PopulationSize):
        a =   list(population[i][0:int(bit/2)])
        b =   list(population[i][int(bit/2):]) 
        x1 = Range[0] + ((Range[1] - Range[0]) / (2**bit - 1)) * BinToDec(a)
        x2 = Range[0] + ((Range[1] - Range[0]) / (2**bit - 1)) * BinToDec(b)
        x.append([x1,x2])
    
    "Find fitness"
    
    fitness = []
    for i in np.arange(PopulationSize):
        fitness.append(RosenBrock(x[i][0],x[i][1]))
    #print("Fitness",fitness)
    
    "Tournament Seletion"
    
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
    
    "Crossover"
    
    CrossOveredExamples = []
    while True:
#        splitJunction = 0
#        while True:
#            if np.random.uniform(0,1) > crossoverProbability or splitJunction == (bit-1):
#                break
#            splitJunction += 1
        
        splitJunction = np.random.randint(bit-1)
        p1 = newSolution[np.random.randint(Range[1])]
        p2 = newSolution[np.random.randint(Range[1])]
    
        if splitJunction >= bit:
            CrossOveredExamples.append(np.append(p1[:splitJunction],p2[splitJunction:]))
        else:  
            CrossOveredExamples.append(np.append(p1[splitJunction:],p2[:splitJunction]))
       
        if len(CrossOveredExamples) == len(newSolution):
            break
    #print("crossover examples",CrossOveredExamples)
    
    "Mutation"
    
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
    
    #print("mutated population",mutatePopulation)
    population = mutatePopulation

a = list(population[np.argmin(fitterSolutions)][0:int(bit/2)])
b = list(population[np.argmin(fitterSolutions)][int(bit/2):])

x1 = Range[0] + ((Range[1] - Range[0]) / (2**bit - 1)) * BinToDec(a)
x2 = Range[0] + ((Range[1] - Range[0]) / (2**bit - 1)) * BinToDec(b)

print(x1,x2)
print(RosenBrock(x1,x2))
