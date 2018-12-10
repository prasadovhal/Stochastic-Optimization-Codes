#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 09:08:40 2018

@author: prasad
@Roll Np. : CMS1731
"""

import numpy as np
import matplotlib.pyplot as plt

A = B = 1.0
C = 0.999
Lk = 1000#int(input("Enter length of markov chain "))
ZetaMin = 600#int(input("Enter minimum number of transitions to be complete "))

def f(R):
    return ((A/R**12) - (B/R**6))

#def SimulatedAnnealing():
x_old = np.random.uniform(0.2,2)
E1 = f(x_old)
T0 = 1000
start = 0
"Initializing T0 for start"
while(True):
    if start == 1:
        x_val = []
        energy = []
    if start == 0:
        m1 = 0
        m2 = 0
        energy = []
        x_val = []
        AcceptedEnergyWithprob = []
    for i in np.arange(Lk):
        h = 0.1 * np.random.uniform(-0.05,0.05)
        x_new = x_old + h
        E2 = f(x_new)
        deltaE = E2 - E1
        if(np.sign(deltaE) == -1):
            x_old = x_new    
            E1 = E2
            x_val.append(x_old)
            energy.append(E1)
            if len(energy) > ZetaMin:
                break
            if start == 0:
                m1 += 1 
        elif(np.sign(deltaE) == 1):
            p = np.exp(-deltaE/T0)
            if(np.random.uniform(0,1) < p):
                x_old = x_new    
                E1 = E2
                x_val.append(x_old)
                energy.append(E1)
                if len(energy) > ZetaMin:
                    break
                if start == 0:
                    m2 += 1
                    AcceptedEnergyWithprob.append(E1)

    if start == 0:
        x = (np.mean(m1) + np.mean(m2) * np.exp(-np.mean(AcceptedEnergyWithprob)/T0))/(np.mean(m1)+np.mean(m2))
        if(x < 0.95):
            T0 += 100
        else:
            print("Temperature is Accepted.....")
            start += 1
            continue
    start = 2
    T0 = C * T0
    print(T0)
    if T0 <= 5:
        break

            
plt.plot(x_val,energy)
plt.show()    
ind = np.argmin(energy)
x_min = x_val[ind]
print(x_min)
