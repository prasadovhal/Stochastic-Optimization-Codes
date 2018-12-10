#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:17:32 2018

@author: prasad
@Roll No. : CMS1731
"""

"""Genetic Algorithm for selecting best parameters for RBF kernel"""

import numpy as np
import pandas as pd
from sklearn import datasets,preprocessing,svm

def BinToDec(x):
    summation = 0
    for i in np.arange(len(x)):
        summation = summation + x[i] * 2**(len(x)-1-i)
    return summation

"SVM Cross Validation Function"
def CrossValidation(c,gm):
    fold = 5
    shuf = np.random.choice(fold,len(data),replace=True)
    accuracy = []
    for i in np.arange(fold):
        "divide data into test and train"
        X_test = data.iloc[np.where(shuf == i)]
        X_train = data.iloc[np.where(shuf != i)]
        y_test = y[np.where(shuf == i)]
        y_train = y[np.where(shuf != i)]
        "Create model for SVM"
        model = svm.SVC(C = c,kernel = "rbf",gamma = gm )
        "fit train data in model"
        model.fit(X_train,y_train)
        "predict test data using fitted model"
        pred = model.predict(X_test)
        "find accuracy for model"
        acc = model.score(X_test,y_test)
        accuracy.append(acc)
    return(np.mean(accuracy))

bit = 12
Range = (1,10)
PopulationSize = 150
population = []

data = pd.DataFrame(preprocessing.scale(datasets.load_iris().data))
y = pd.factorize(datasets.load_iris().target)[0]


for i in np.arange(PopulationSize):
    population.append(np.random.randint(low = 0,high = 2,size = 2*bit))

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
        fitness.append(CrossValidation(x[i][0],x[i][1]))
    
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
        splitJunction = np.random.randint(bit-1)
        p1 = newSolution[np.random.randint(Range[1])]
        p2 = newSolution[np.random.randint(Range[1])]
    
        if splitJunction >= bit:
            CrossOveredExamples.append(np.append(p1[:splitJunction],p2[splitJunction:]))
        else:  
            CrossOveredExamples.append(np.append(p1[splitJunction:],p2[:splitJunction]))
       
        if len(CrossOveredExamples) == len(newSolution):
            break
    
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
    
    population = mutatePopulation

a = list(population[np.argmin(fitterSolutions)][0:int(bit/2)])
b = list(population[np.argmin(fitterSolutions)][int(bit/2):])

x1 = Range[0] + ((Range[1] - Range[0]) / (2**bit - 1)) * BinToDec(a)
x2 = Range[0] + ((Range[1] - Range[0]) / (2**bit - 1)) * BinToDec(b)

print(x1,x2)
print(CrossValidation(x1,x2))
