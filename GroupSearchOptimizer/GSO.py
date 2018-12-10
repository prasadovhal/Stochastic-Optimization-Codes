#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:17:16 2018

@author: prasad
"""
"""Group Search Optimizer"""
import numpy as np

"minimize f(x)"
def f(x):
    x1 = x[0]
    x2 = x[1]
    return (4 * x1**2 - 2.1 * x1**4 + (1/3) * x1**6 + x1 * x2 - 4 * x2**2 + 4 * x2**4)

def direction(phi,n):
    d1 = np.cos(phi)


#fmin = −1.0316285
groupSize = 30
n = 2
Range = (-5,5)

lmax = 0
for i in range(2):
    lmax += (Range[1] - Range[0])**2

lmax = np.sqrt(lmax)

a = np.sqrt(n+1)
thetamax = np.pi / a**2
alphamax = thetamax / 2.0
phi = (np.pi/4) *  np.ones(n)

xinit = np.random.uniform(Range[0],Range[1],size=(groupSize,n))
fitness = []

for i in range(groupSize):
    fitness.append(f(xinit[i]))
    
producer = xinit[np.argmin(fitness)]

"Producer optimization path"
x1 = producer + np.random.uniform(0,1) * lmax * 1