# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

data = pd.read_csv('iris.csv')
print(data.shape)
data.head()

independent = data['petal_length'].values
dependent = data['petal_width'].values

#independent = data['sepal_length'].values
#dependent = data['sepal_width'].values

mean_x = np.mean(independent)
mean_y = np.mean(dependent)

n = len(independent)

numer = 0
denom = 0

#y =  mx + c

for i in range(n):
    numer += (independent[i] - mean_x) * (dependent[i] - mean_y)
    denom += (independent[i] - mean_x) ** 2
    
gradient =  numer / denom
c = mean_y - (gradient * mean_x)

#plt.plot(independent,dependent)

print(gradient, c)

#coba plot

max_x = np.max(independent) + 100
min_x = np.min(independent) - 100

independent_baru = np.linspace(min_x, max_x, 1000)
dependent_baru = c + gradient * independent_baru

#menampilkan grafik data dengan nilai regresinya
plt.plot(independent_baru, dependent_baru, color='#58b970', label = 'Regression Line')
plt.scatter(independent, dependent, color='#ef5423', label='Scatter Plot')
            
#membuat label
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


independent = independent.reshape((n,1))

#creating model
reg = LinearRegression()
reg = reg.fit(independent, dependent)

dependentpred = reg.predict(independent)

r2_score = reg.score(independent, dependent)
print(r2_score)
