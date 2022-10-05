# Implementation of a Multi-Layer Perceptron object
# Foundations of Neural & Cognitive modelling, Lab 4
# 27-02-2020

import numpy as np
import matplotlib.pyplot as plt

class MLP():
    """
    A multi-layer perceptron with 1 hidden layer and sigmoid activation function.
    """
    def __init__(self, n_input, n_hidden, n_output):
        """
        Initializes the MLP.
        
        Input:
          n_input -- number of input nodes
          n_hidden -- number of hidden nodes
          n_output -- number of output nodes
        """
        # initialize weights and bias from input to hidden layer
        self.weights_h = np.random.uniform(low=-1., high=1., 
                                           size=(n_hidden, n_input))
        self.bias_h = np.random.uniform(low=-1., high=1., 
                                        size=n_hidden)
        
        # initialize weights and bias from hidden to output layer
        self.weights_o = np.random.uniform(low=-1., high=1., 
                                           size=(n_output, n_hidden))
        self.bias_o = np.random.uniform(low=-1., high=1., 
                                        size=n_output)
                 
    def sigm(self, x):
        """
        Sigmoid activation function.
        """
        return 1./(1. + np.exp(-x))
                 
    def dsigm(self, sigmz):
        """
        Derivative of the sigmoid activation function.
        
        Input:
          sigmz -- layer activations: sigmoid(z)
        """
        return sigmz * (1. - sigmz)
        
    def feedforward(self, input_x):
        """
        Performs a feedforward pass through the network on a
        single input vector.
        
        Input:
          input_x -- the input vector (X[i,])
          
        Output:
          h -- the hidden layer activation
          o -- the output layer activation (prediction)
        """
        
        # compute hidden layer activations
        z1 = self.weights_h @ input_x + self.bias_h
        h = self.sigm(z1)
                 
        # compute output layer activations
        z2 = self.weights_o @ h + self.bias_o
        o = self.sigm(z2)
        
        return h, o
                 
    def predict(self, input_x):
        """
        Performs a feedforward pass and return the MLP's prediction
        on a single input vector.
        
        Input:
          input_x -- the input vector (X[i,])
          
        Output:
          o -- the prediction (y-hat, output of the MLP)
        """
        
        # perform feedforward pass
        h, o = self.feedforward(input_x)
                 
        return o
        
    def backpropagation(self, input_x, target, prediction, h, learn_rate):
        """
        Backpropagates error through the network and computes updates to
        weights and bias terms.
        
        Input:
          input_x -- the input vector ([X[i,]])
          target -- the target (y[i])
          prediction -- the prediction made by the network (o, y-hat)
          h -- the hidden layer activations
          learn_rate -- the learning rate
          
        Output:
          weights_h_change -- update value for the hidden layer weights
          bias_h_change -- update value for the hidden layer bias term
          weights_o_change -- update value for the output layer weights
          bias_o_change -- update value for the output layer bias term
        """
        
        # compute output layer delta
        delta_o = (target - prediction) * self.dsigm(prediction)

        # compute update for output layer weights and bias
        weights_o_change = learn_rate * np.outer(delta_o, h)
        bias_o_change = learn_rate * delta_o

        # compute hidden layer delta
        delta_h = (self.weights_o.T @ delta_o) * self.dsigm(h)

        # compute update for hidden layer weights and bias
        weights_h_change = learn_rate * np.outer(delta_h, input_x)
        bias_h_change = learn_rate * delta_h
                 
        return weights_h_change, bias_h_change, \
                 weights_o_change, bias_o_change
                 
    def train(self, X, y, learn_rate=0.1, maxit=100):
        """
        Iterates over a dataset and trains the MLP.
        
        Input:
          X -- (N,D) array in which each row is a data point
          y -- an (N, 1) array in which each row is the target of X[i,]
          learn_rate -- the learning rate
          maxit -- number of iterations over the dataset
          
        Output:
          errors -- the error over the training set at each iteration
        """
        
        errors = np.zeros(maxit)   
        
        for it in range(maxit):
            for i in range(len(X)):
                # pick training example
                example_input = X[i,]
                example_target = y[i,]

                # feedforward step
                h, o = self.feedforward(X[i,])

                # backpropagation
                weights_h_change, bias_h_change, \
                weights_o_change, bias_o_change = \
                self.backpropagation(example_input, example_target, 
                                     o, h, learn_rate)

                # update weights
                self.weights_h += weights_h_change
                self.weights_o += weights_o_change

                # update biases
                self.bias_h += bias_h_change
                self.bias_o += bias_o_change
                
            errors[it] = self.compute_error(X, y)
        return errors
        
    def compute_error(self, X, y):
        """
        Compute the weighed sum of squared errors (SSE) over the dataset.
        
        Input:
          X -- (N,D) array in which each row is a data point
          y -- an (N, 1) array in which each row is the target of X[i,]
          
        Output:
          weighed_sse -- the weighed SSE over all datapoints
        """
        predictions = np.array([self.predict(X[i,]) for i in range(len(X))])
        weighed_errors = np.zeros(len(predictions))
        
        for i in range(len(predictions)):
            squared_error = (y[i] - predictions[i])**2
            class_size = np.sum(y == y[i])
            weighed_errors[i] = squared_error / class_size
            
        weighed_sse = np.sum(weighed_errors)
        
        return weighed_sse
    
def plot_errors(errors, ylabel):
    plt.plot(errors)
    plt.xlim(0, len(errors))
    plt.ylim(0, 1)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.ylabel(ylabel, size=15)
    plt.xlabel('Iteration', size=15)
    plt.show()