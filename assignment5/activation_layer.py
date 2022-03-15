from layer import Layer

class ActivationLayer(Layer):

    def __init__(self, activation, dactivation):
        self.activation = activation
        self.dactivation = dactivation

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, out_error, lr):
        return self.dactivation(self.input) * out_error

