import numpy as np

# activation function and its derivative

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1-np.tanh(x)**2

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1- sigmoid(x))