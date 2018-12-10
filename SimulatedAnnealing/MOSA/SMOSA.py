#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 11:14:42 2018

@author: prasad
@Roll No : CMS1731
"""

""" SMOSA """

import numpy as np
import matplotlib.pyplot as plt

"Minimize both objective function"
def obj1(x):
    return x**2

def obj2(x):
    return (x-2)**2

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
   
"Initialize temprature"     
T = [1000,1000]    
E = []
x = []
"Initialize Pareto set"
ParetoSet = []
"Start with random value"
x_old = np.random.uniform(low = -5, high = 5)
ParetoSet.append(x_old)
"Energy at start point"
E1 = [obj1(x_old) , obj2(x_old)]

"Temperature loop"
while T[0] > 10:
    print(T[0])
    "Iterative loop"
    for j in range(100):
        "Perturbation"
        h =  np.random.uniform(-1,1)
        "New point"
        x_new = x_old + h
        "Energy at new point"
        E2 = [obj1(x_new) , obj2(x_new)]
        "Update Pareto Set"
        ParetoSet = ParetoSetUpdate(ParetoSet,x_new)
        "Acceptance with probability"
        deltaE = 1
        for i in range(len(E1)):
            deltaE *= np.exp(-(E1[i] - E2[i]) / T[i])
            
        if np.random.uniform(0,1) < deltaE:
            x_old = x_new
            E1 = E2
            E.append(E1)
            x.append(x_old)
    "Starting with other point from Pareto set randomly"
    x_old = np.random.randint(len(ParetoSet))
    "Temperature Schduling"
    T = [ 0.9 * T[0] , 0.9 * T[1] ]
    

print("Pareto set is " , ParetoSet)
print("Energies at Pareto set are : ")
final = []
for i in range(len(ParetoSet)):
    final.append([obj1(ParetoSet[i]) , obj2(ParetoSet[i])])
print(final)

plt.plot(ParetoSet,np.array(final)[:,0])
plt.plot(ParetoSet,np.array(final)[:,1])
plt.show()
