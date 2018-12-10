#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:39:46 2018

@author: prasad
"""

import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt

def RosenBrock(x):
   return ((1.0 - x[0])**2 + (100 * (x[1] - x[0]**2)**2))

def SMC():
    x_old = np.random.uniform(-5,5,size=2)
    E1 = RosenBrock(x_old)
    energy = []
    x_val = []
    for i in np.arange(1000):
        h = 0.1 *np.random.uniform(-1,1,size=2)
        x_new = x_old + h
        E2 = RosenBrock(x_new)
        if(np.sign(E2-E1) == -1):
            x_old = x_new    
            E1 = E2
            x_val.append(x_old)
            energy.append(E1)
    return x_val , np.array(energy)

X , E = SMC()
plt.hist(E)
#"Histogram estimation simple method"
#H1 = -sum(E * np.log(E)) / max(E)
#"Rice Rule"
#w = 2*len(E)**(1/3)
#"Histogram estimation for uniform bin width"
#H2 = -sum(E * np.log(E/w))

"Histogram estimation"
h = (4*np.std(E)**5 / (3 * len(E)))**(1/5)