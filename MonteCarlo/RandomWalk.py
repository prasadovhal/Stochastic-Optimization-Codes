#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 22:35:55 2018

@author: prasad
@Roll No. CMS1731
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 4 * x - 4

""""problem = input("enter funtion has to either maximize or minimize:")"""
x_old = np.random.random()
E1 = f(x_old)
x = []
energy = []
for i in np.arange(100):
    h = 0.1 * np.random.uniform()
    x_new = x_old + h
    E2 = f(x_new)
    x_old = x_new    
    print("x is %f and energy is %f"%(x_old,E2))
    x.append(x_new)
    energy.append(E2)

plt.plot(x,energy)
plt.show()
ind = np.argmin(energy)
x_min = x[ind]
print(x_min)
