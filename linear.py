# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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


# 5
# 5.1 Split-out validation dataset
array = data.values
X = array[:,0:4]        # list slicing for attributes. [start:stop:step], def step = 1. in this case [from start:until last instance (,0 until last first col):step = 4 (4 columns to copy and skip last column)]
Y = array[:,4]          # list slice for class column
validation_size = 0.20  # 20% for the validation set
seed = 7
# returns tuple of values
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed) # understand what thsis does mroe indepth and write about it. doing some fitting, look at leetcode intro to ML
# Split arrays or matrices into random train and test subsets. (numpy arr, numpy arr, testsize is the portion of the dataset to use for the test split, random_state is Pseudo-random number generator state used for random sampling.)
# X_train is for the intances used for training
# Y_train is for the expected outcome of each instance
# X_validation is the instances used for validating the model
# Y_validation is for the expected outcome of each corresponding instance in X_validation

#5.2 Test harness
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# 5.3 build model
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed);                                          # sklearn. KFold provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (>= 2 folds) (without shuffling by default). No shuffle to compare algorithms
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring);       # Evaluate a score by cross validation (model is the algorithm to fit the data,x_train is the data to fit for the model,y_train is target variable to predict; result of the x_train instances,cv is the cross validation splitting strat,scoring is the accuracy of the test set)
	results.append(cv_results);                                                                             # store the scores (array) of each run of the cross validation in the result array
	names.append(name);                                                                                     # Stores the name of the algorithm for the current result
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(cv_results);
	print(msg)

	
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# 6 Make predictions on validation dataset
print("Predicting on unseen data.");
knn = KNeighborsClassifier();                                   # sklearn lib. stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point
knn.fit(X_train, Y_train);                                      # train the model with the train dataset
predictions = knn.predict(X_validation);                        # get the predictions using the validation test with this model knn
print(accuracy_score(Y_validation, predictions));               # compares the validation known answer with the predicted to determine accuracy
print(confusion_matrix(Y_validation, predictions));             # matrix of accuracy classification where C(0,0) is true negatives, C(1,0) is false negatives, C(1,1) true posivtes, C(0,1) false positvies.
print(classification_report(Y_validation, predictions));        # text report
print("X_validation predict ===");
print(predictions);                                             # array of predicted values

for row_index, (input, predictions, Y_validation) in enumerate(zip (X_validation, predictions, Y_validation)):
  if predictions != Y_validation:
    print('Row', row_index, 'has been classified as ', predictions, 'and should be ', Y_validation)
    print(X_validation[row_index]);

# done tutorial, so lets manually put some unseen data and see what the output is.
print("Manaul input of unseen data to validate to me =====");
x_manual = [[5.1, 3.5,1.4,0.2]];
manualPred = knn.predict(x_manual);
print(manualPred);