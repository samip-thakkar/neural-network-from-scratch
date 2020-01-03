# -*- coding: utf-8 -*-
"""

@author: Samip
"""

#CREATING A NEURAL NETWORK CLASS

#Import libraries
import numpy as np
from scipy.optimize import minimize
from sigmoid import sigmoid, sigmoid_grad

#Main class
class NeuralNetwork:
    
    #Constructor    
    def __init__(self, total_layers, total_nodes, rglr_param):
        
        """
        Arguments: 
        total_layers (int): Total number of layers of NN (I/p + O/p)
        total_nodes (list): Number of nodes in each hidden layer (Bias is not included)
        rglr_param (float): Regularization Parameter
        
        Attributes:
        total_layers (int): Total number of layers of NN (I/p + O/p)
        total_nodes (list): Number of nodes in each hidden layer (Bias is not included)
        rglr_param (float): Regularization Parameter
        """
    
        self.total_layers = total_layers
        self.total_nodes = np.array([0] + total_nodes + [0])
        self.rglr_param = rglr_param
        
    
    #Feed Forward function to calculate the value of output layer for given input and weights
    def feedforward(self, X, weights):
        
        """
        Arguments:
            X (2darray): A m*n matrix representing m inputs with n features
            weights (list): A list consisting weights of each layers
            
        Return tyeps:
            node_Z (list): A list consists of node values of each layer
            node_A (list): A list that contains the activated node values of each layer (Bias is added to all except output layer)
        """
        
        node_z = [X]
        node_a = [X]
        
        for weight in weights:
            node_a[-1] = np.insert(node_a[-1], 0, 1, axis = 1) #(add 1 to 0th column)
            node_z.append(node_a[-1] @ weight.T)
            node_a.append(sigmoid(node_z[-1]))
        return node_z, node_a
    
    
    #Cost Function
    def cost(self, y_pred, y, weights):
        
        """
        Arguments:
            y_pred (2darray): Predicted output by neural network
            y (2darray): Actual output of the neural network
            
        Return:
            cost (float): Value of cost function
        """
        
        m = y.shape[0]
        
        cost = sum(np.einsum('ij,ij->i', -y, np.log(y_pred))
               - np.einsum('ij,ij->i', 1-y, np.log(1 - y_pred)))
        cost += (self.rglr_param / 2
                 * sum([np.sum(weight[: , 1:] ** 2) for weight in weights]))
        cost /= m
        return cost
    
    #Backpropogation to calculate the gradient of the cost function to minimize it
    def backpropogation(self, weights, X, y):
        
        """
        Arguments:
            weights (1darray): Weights of each layer
            X (2darray): A m*n matrix repesenting training input with n features
            y (2darray): A m*K matrix representing training output in K classes
            
        Return:
            cost (float): Value of cost function
            grad_unrolled (1darray): An unrolled gradient of the cost function
        """
        
        m = y.shape[0]
        weights = self.reshape_weights(weights)
        
        #Feedforward function
        node_z, node_a = self.feedforward(X, weights)
        
        #Cost Function
        cost = self.cost(node_a[-1], y, weights)
        
        #Compute the gradient 
        delta = [node_a[-1] - y]
        delta[-1] = np.insert(delta[-1], 0, 1, axis=1)
        node_a.pop()
        node_z.pop()
        grads = []
        while weights:
            grads.append(delta[-1][:, 1:].T @ node_a[-1])
            grads[-1][:, 1:] += self.rglr_param * weights[-1][:, 1:]
            node_z[-1] = np.insert(node_z[-1], 0, 1, axis=1) 
            delta.append((delta[-1][:, 1:] @ weights[-1])
                         * sigmoid_grad(node_z[-1]))
            node_a.pop()
            node_z.pop()
            weights.pop()
        
        # Unroll the list of gradients into 1darray
        grad_unrolled = np.array([])
        for grad in reversed(grads):
            grad_unrolled = np.concatenate((grad_unrolled, 
                                            np.ravel(grad)))
       
        # Finish the computaion of the gradient
        grad_unrolled /= m
        
        return cost, grad_unrolled
    
    #Initialize weight parameter randomly
    def rand_init_weights(self):
     
        """
        Returns:
            init_weights (1darray): An unrolled initial weight parameter for each layer
        """
        
        def init_epsilon(nodes_in, nodes_out):
            return np.sqrt(6) / np.sqrt(nodes_in+nodes_out)
        
        # Randomly initialize weight between an suitable interval for each layer 
        init_weights = np.array([])
        for nodes_in, nodes_out in list(zip(self.total_nodes[:-1], 
                                            self.total_nodes[1:])):
            epsilon = init_epsilon(nodes_in, nodes_out)
            rand_weight = (2*epsilon * (np.random.random(nodes_out*(nodes_in+1)))
                           - epsilon)
            init_weights = np.concatenate((init_weights, rand_weight))
            
        return init_weights
    
    #Reshape an unrolled weight parameter into a list of weights
    def reshape_weights(self, weights_unrolled):
        
        """
        Arguments:
            weights_unrolled (1darray): The unrolled weight to be reshaped
        
        
        Returns:
            weights (list): A list that consists of weight for each layer
        """
        
        weights = []
        for nodes_in, nodes_out in list(zip(self.total_nodes[:-1], 
                                            self.total_nodes[1:])):
            weight = np.reshape(weights_unrolled[:nodes_out*(nodes_in+1)], 
                                (nodes_out, nodes_in+1))
            weights.append(weight)
            weights_unrolled = np.delete(weights_unrolled, 
                                         np.s_[:nodes_out*(nodes_in+1)])
        return weights
    
    #Training the neural networks
    def train(self, X, y):
        
        """
        Arguments:
            X (2darray): An mxn array representing m training inputs with n features
            y (2darray): An mxK array representing m training outputs in K classes
        
        
        Return:
            Trained_weights (list): A good set weights that minimizes the cost function
        """
        
        self.total_nodes[0] = X.shape[1]
        self.total_nodes[-1] = y.shape[1]
        
        # Initialize weights
        init_weights = self.rand_init_weights()
        
        # Find weights that minimize the cost function
        min_result = minimize(fun=self.backpropogation, x0=init_weights, args=(X, y),
                              method='TNC', jac=True, options={'maxiter': 300})
        trained_weights = self.reshape_weights(min_result.x)
        return trained_weights
  
    
    #Prediction function to predict using Neural network model
    def predict(self, weights, X):
        """
        Arguments:
            weights (list): A list that consists of trained weight for each layer
            X (2darray): An mxn array representing m test inputs with n features
        
        
        Returns:
            pred_y (2darray): The predicted output
        """
        
        node_z, node_a = self.feedforward(X, weights)
        y_pred = node_a[-1]
        return y_pred
    
    