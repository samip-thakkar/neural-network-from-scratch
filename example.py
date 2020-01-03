# -*- coding: utf-8 -*-
"""

@author: Samip
"""
#Import libraries and the main Neural Network Class
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.io import loadmat
import neuralnetwork

#Load the data
data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y)

#Predict the output using neural network model
nn = neuralnetwork.NeuralNetwork(3, [25], 1)
weights = nn.train(X, y_encoded)
y_pred = nn.predict(weights, X)
y_pred = np.argmax(y_pred, axis=1) + 1

#Print the accuracy
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))