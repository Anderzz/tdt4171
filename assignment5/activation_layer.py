from layer import Layer

class ActivationLayer(Layer):

    def __init__(self, activation, dactivation):
        self.activation = activation
        self.dactivation = dactivation

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, out_error, lr=0.01):
        return self.dactivation(self.input) * out_error

