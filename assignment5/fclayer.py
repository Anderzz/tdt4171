from layer import Layer
import numpy as np

class FCLayer(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5


    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, out_error, lr=0.01):
        input_error = np.dot(out_error, self.weights.T)
        weights_error = np.dot(self.input.T, out_error)
        bias_error = out_error

        #update params
        self.weights -= lr * weights_error
        self.bias -= lr * bias_error
        return input_error
