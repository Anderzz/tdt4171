from layer import Layer
import numpy as np

class FCLayer(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    #forward pass
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, out_error, lr):
        input_error = np.dot(out_error, self.weights.T)
        weights_error = np.dot(self.input.T, out_error)
        bias_error = out_error

        #SGD
        self.weights -= lr * weights_error
        self.bias -= lr * bias_error
        #pass the gradients on
        return input_error
