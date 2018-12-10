#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 10:01:01 2018

@author: prasad
@Roll No. : CMS1731
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 22:32:26 2018

@author: prasad
@Roll No. CMS1731
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing,datasets,svm

def f(C,gamma):
    data = pd.DataFrame(datasets.load_breast_cancer().data)
    y = datasets.load_breast_cancer().target
    
    data = pd.DataFrame(preprocessing.scale(data))
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
        model = svm.SVC(C = C,kernel = "rbf",gamma = gamma)
        "fit train data in model"
        model.fit(X_train,y_train)
        "predict test data using fitted model"
        pred = model.predict(X_test)
        "find accuracy for model"
        acc = model.score(X_test,y_test)
        accuracy.append(acc)
    return(np.mean(accuracy))


counterC = 0
counterGamma = 0
zeta = 0.999
C = [0.01,1,10,100,1000,10000]
gamma = 1/np.array([0.01,1,10,100,1000,10000])

E1 = f(C[0],gamma[0])

temp = 1000
energy = []
x_val = []
setCGamma = []
while(True):
    for i in np.arange(1,len(gamma)-1):
        E2 = f(C[counterC],gamma[i])
        deltaE = E2 - E1
        if(np.random.uniform(0,1) < np.exp(deltaE/temp)):
            E1 = E2
            energy.append(E1)
            setCGamma.append([C[counterC],gamma[i]])
    temp = zeta * temp
    counterC += 1
    if counterC == len(C):
        break
    print(temp)

ind =  np.argmax(energy)
minCnGamma = setCGamma[ind]
print(minCnGamma)