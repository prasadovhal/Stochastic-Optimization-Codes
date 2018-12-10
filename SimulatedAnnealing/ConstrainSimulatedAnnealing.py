#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sept 24 22:32:26 2018

@author: prasad
@Roll No. CMS1731
"""
import numpy as np
import matplotlib.pyplot as plt

def ObjFunction(x1,x2,x3):
    f = (x3 + 2) * x1**3 * x2
    return f
    
def inequalityConstraints(x1,x2,x3):
    g1 = 1 - ((x2**3 * x3) / (71785 * x1**4))
    g2 = 1 - ((140.45 *  x1) / (x2**2 * x3))
    g3 = ((x1 + x2)/1.5) - 1
    g4 = ((4*x2**2 - x1*x2) / (12566 *(x1**3 * x2 - x1**4))) + (1 / (5108 * x1**2)) - 1
    Sum = [max(0,g1) , max(0,g2) , max(0,g3) , max(0,g4)]
    return sum(Sum)

def fitness(x1,x2,x3):
    C = 0.01
    return (ObjFunction(x1,x2,x3) + C * inequalityConstraints(x1,x2,x3))

NumOfVar = 3
x1_old, x2_old, x3_old = np.random.uniform(-2,2,size = NumOfVar)
E1 = fitness(x1_old, x2_old, x3_old)

temp = 1000
energy = []
x_val = []
x_val.append([x1_old, x2_old, x3_old])
while(temp >= 1):
    for i in np.arange(10):
        h = 0.1 * np.random.uniform(-0.5,0.5,size = NumOfVar)
        x1,x2,x3 = [x1_old, x2_old, x3_old] + h
        E2 = fitness(x1,x2,x3)
        if(np.random.uniform(0,1) < np.exp(-(E2-E1)/temp)):
            x1_old, x2_old, x3_old = x1, x2, x3
            E1 = E2
            x_val.append([x1_old, x2_old, x3_old])
            energy.append(E1)
    temp = 0.999 * temp
    print(temp)

#plt.plot(x_val,energy)
#plt.show()    
#ind = np.argmin(energy)
#x_min = x_val[ind]
#print(x_min)
