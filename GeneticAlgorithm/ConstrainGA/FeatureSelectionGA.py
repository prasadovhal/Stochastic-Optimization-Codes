#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 14:49:20 2018

@author: prasad
@Roll No.: CMS1731
"""
"""Genetic Algorithm for Feature Selection using constrains"""

"Import Libraries"
import numpy as np
import pandas as pd
from sklearn import datasets,preprocessing,svm
from sklearn.metrics import confusion_matrix

"SVM Cross Validation Function"
def CrossValidation(data,y):
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
        model = svm.SVC(C = 1.0,kernel = "rbf")
        "fit train data in model"
        model.fit(X_train,y_train)
        "predict test data using fitted model"
        pred = model.predict(X_test)
        "find accuracy for model"
        Acc = model.score(X_test,y_test)
        "Confusion matrix"
        cm = confusion_matrix(y_test,pred)
        "constrains: penalty for wrongly predicted examples"
        wrongPred = (sum(sum(cm)) - sum(np.diag(cm))) / len(y_test)
        acc = Acc - wrongPred
        accuracy.append(acc)
    return(np.mean(accuracy))
    
"Removing population with all zeros i.e. no attributes selected"
def CheckforNullPopulation(population):
    i = 0
    while(i < len(population)):
        if sum(population[i]) == 0:
            population.pop(i)
        i += 1
    return population

"Tournament Selection function"
def TournamentSelection(fitness,population,newPopulationSize):
    fitterSolutions = []
    while True:
        p1 = fitness[np.random.randint(newPopulationSize)]
        p2 = fitness[np.random.randint(newPopulationSize)]
        
        if p1 >= p2:
            fitterSolutions.append(p1)
        
        if len(fitterSolutions) == len(fitness):
            break
    
    fitterSolutionsIndex = []
    for i in np.arange(len(fitterSolutions)):
        fitterSolutionsIndex.append(np.where(fitterSolutions[i] == fitness)[0][0])
    
    newSolution = []
    for i in np.arange(len(fitterSolutionsIndex)):
        newSolution.append(population[fitterSolutionsIndex[i]])
    return fitterSolutions , newSolution

"Crossover function"
def Crossover(newSolution,bitSize,newPopulationSize):
    CrossOveredExamples = []
    while True:        
        splitJunction = np.random.randint(bitSize-1)
        p1 = newSolution[np.random.randint(newPopulationSize)]
        p2 = newSolution[np.random.randint(newPopulationSize)]
        
        if splitJunction >= bitSize:
            CrossOveredExamples.append(np.append(p1[:splitJunction],p2[splitJunction:]))
        else:  
            CrossOveredExamples.append(np.append(p1[splitJunction:],p2[:splitJunction]))
        
        CrossOveredExamples = CheckforNullPopulation(CrossOveredExamples)
        if len(CrossOveredExamples) == len(newSolution):
            break
    return CrossOveredExamples

"Mutation function"
def Mutation(CrossOveredExamples,newPopulationSize,bitSize,mutationProbability,newSolution):
    mutatePopulation = []
    while True:
        mutationExample = CrossOveredExamples[np.random.randint(newPopulationSize)]
        flip = []
        for i in np.arange(bitSize):
            if np.random.uniform(0,(mutationProbability+0.01)) < mutationProbability:
                flip.append(abs(mutationExample[i] - 1))
            else:
                flip.append(mutationExample[i])
        mutatePopulation.append(np.array(flip))
        mutatePopulation = CheckforNullPopulation(mutatePopulation)
        if len(mutatePopulation) == len(newSolution):
            break     
    return mutatePopulation

"Import datasets"
data = pd.DataFrame(preprocessing.scale(datasets.load_iris().data))
y = pd.factorize(datasets.load_iris().target)[0]

"Best subset and best accuracy finding function"
def BestSubset(data,y,population):
    fitness = []
    for i in range(len(population)):
        SampleData = (data.drop((np.where(population[i]==0)[0]),axis=1))
        fitness.append(CrossValidation(SampleData,y))

    BestSubsetAccuracy = max(fitness)
    bestSubset = population[np.where(BestSubsetAccuracy == fitness)[0][0]]
    return bestSubset,BestSubsetAccuracy

"Main function"
def GA():
    "Define Parameters"
    PopulationSize = 5
    bitSize = data.shape[1]
    population = []
    crossoverProbability = 0.7
    mutationProbability = 1 / PopulationSize
    
    "Creating population"
    for i in np.arange(PopulationSize):
        population.append(np.random.randint(low = 0,high = 2,size = bitSize))
    
    population = CheckforNullPopulation(population)
    newPopulationSize = len(population)
    #print(newPopulationSize)
    
    iter = 0
    "Creating generations"
    for k in range(10):
        iter += 1
        print(iter)
        "Finding fitness ; here it is cv-accuracy using SVM"
        fitness = []
        for i in range(len(population)):
            SampleData = (data.drop((np.where(population[i]==0)[0]),axis=1))
            fitness.append(CrossValidation(SampleData,y))
        
        "Tournament Selection"    
        fitterSolutions , newSolution = TournamentSelection(fitness,population,newPopulationSize)
        
        "Crossover"
        CrossOveredExamples = Crossover(newSolution,bitSize,newPopulationSize)
        
        "Mutation"
        mutatePopulation = Mutation(CrossOveredExamples,newPopulationSize,bitSize,mutationProbability,newSolution)
            
        population = mutatePopulation
    
    "Finding Best Subset"
    bestSubset, BestSubsetAccuracy = BestSubset(data,y,population)
    return bestSubset, BestSubsetAccuracy

bestSubset , BestSubsetAccuracy = GA()
print("Best subset is ",bestSubset ,"and corrosponding accuracy is" ,BestSubsetAccuracy)
