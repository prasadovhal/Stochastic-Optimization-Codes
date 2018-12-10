#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 22:32:26 2018

@author: prasad
@Roll No. CMS1731
"""
import numpy as np
import matplotlib.pyplot as plt

A = B = 1.0

def f(R):
    return ((A/R**12) - (B/R**6))

def SimulatedAnnealing():
    x_old = np.random.uniform(0.2,2)
    E1 = f(x_old)
    
    temp = 1000
    energy = []
    x_val = []
    x_val.append(x_old)
    energy.append(E1)
    while(temp >= 1):
        for i in np.arange(100):
            h = 0.1 * np.random.uniform(-0.05,0.05)
            x_new = x_old + h
            E2 = f(x_new)
            if(np.random.uniform(0,1) < np.exp(-(E2-E1)/temp)):
                x_old = x_new    
                E1 = E2
                x_val.append(x_old)
                energy.append(E1)
                #print("x is %f and energy is %f"%(x_new,E2))
        temp = 0.999*temp
        print(temp)
    
    plt.plot(x_val,energy)
    plt.show()    
    ind = np.argmin(energy)
    x_min = x_val[ind]
    print(x_min)
