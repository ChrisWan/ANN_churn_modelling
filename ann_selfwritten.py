# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:51:14 2020

@author: wanke
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)

X = ct.fit_transform(X)
# old code, found above one
onehotencoder = OneHotEncoder(categories = [1])
X = onehotencoder.fit_transform(X).toarray()
# end old code
# Dummy variable - delete first column
X = X[:,1:]

# Change the package name!!!
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create the ANN here
# Import libaries here
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# Tip for number of neurons = 1+number of features(11)/2; relu = rectifier function and for output activation function sigmoid
# Had to change some parameter names and do not know about input_dim = how many inputs to expect = 11 independent variables = 11 features
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
classifier.add(Dropout(p = 0.1)) # add dropout with 10% to first hidden layer
# Adding a second hidden layer; input_dim not needed because not the first layer and number of neuron of previous layer is known
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1)) # to the first second layer
# Adding the final layer = output layer; if more than just yes/no units=number of categories; activation = 'softmax'
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN; add gradient descent; loss function binary_crossentropy = for binary results; if more than 2 categories -> categorical_crossentropy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set; batch_size = after how many you want to update the weights; epochs = how many rounds of training
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results; for the confusion matrix we need true or false values = transform numbers with a threshold into true/false
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Make single prediction for one customer; have to put information in a horizontal vector; when 0.0 all parameteres are typecasted to float instead of integers
new_customer_prediction = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_customer_prediction = (new_customer_prediction > 0.5)
new_customer_prediction

# Evaluating the ANN model; create own build_classifier function and evalute "real" performance = accuracy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# create a new global classifier variable and it will not be trained on the whole training set
# it will be trained on the single k-folds
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
# create accuracies variables; for cv is 10 a good number; n_jobs = -1 use all cpus
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()
print('mean = {:f} and variance = {:2f}'.format(mean,variance))

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameteres = {'batch_size': [15,16], 'epochs': [600,650],
               'optimizer': ['rmsprop']}

grid_search = GridSearchCV(estimator = classifier, param_grid = parameteres, scoring = 'accuracy', cv = 10, n_jobs =-1)
grid_search = grid_search.fit(X_train, y_train)

best_parameter = grid_search.best_params_ 
best_accuracy = grid_search.best_score_





















